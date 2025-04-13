import { useState, useEffect } from 'react'
// import reactLogo from './assets/react.svg'
// import viteLogo from '/vite.svg'
import './App.css'
import GenerationTab from './components/GenerationTab'; // Assuming component path
import IdeationTab from './components/IdeationTab';     // Assuming component path

function App() {
  // --- State Variables ---

  // Tab Management
  const [activeTab, setActiveTab] = useState('Ideation'); // Start on Ideation tab

  // Generation State (moved from top-level)
  const [scriptSegment, setScriptSegment] = useState("");
  const [generalContext, setGeneralContext] = useState("");
  const [previousCodeAttempt, setPreviousCodeAttempt] = useState("");
  const [enhancementRequest, setEnhancementRequest] = useState("");
  const [finalCommand, setFinalCommand] = useState("");
  const [sceneName, setSceneName] = useState("");
  const [saveGeneratedCode, setSaveGeneratedCode] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false); // Renamed from isLoading
  const [generationOutput, setGenerationOutput] = useState(""); // Renamed from responseOutput
  const [errorLoadingDefaults, setErrorLoadingDefaults] = useState("");

  // Ideation State
  const [ideationInputText, setIdeationInputText] = useState("");
  const [ideationOtherInfo, setIdeationOtherInfo] = useState("");
  const [ideationInitialIdea, setIdeationInitialIdea] = useState("");
  const [numIdeas, setNumIdeas] = useState(3); // Added state, default 3
  const [generatedIdeas, setGeneratedIdeas] = useState([]);
  const [isIdeating, setIsIdeating] = useState(false);
  const [ideationError, setIdeationError] = useState("");


  // --- Effects ---

  // Fetch default configuration on component mount (for Generation Tab)
  useEffect(() => {
    const fetchDefaults = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/config/defaults/manim');
        if (!response.ok) {
          throw new Error(`Failed to fetch defaults: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        setGeneralContext(data.defaultGeneralContext || "");
        setFinalCommand(data.defaultFinalCommand || "");
        setErrorLoadingDefaults("");
      } catch (error) {
        console.error("Error fetching config defaults:", error);
        setErrorLoadingDefaults(`Error loading default prompts: ${error.message}. Using empty defaults.`);
        // Don't reset here, allow existing values if defaults fail
        // setGeneralContext("");
        // setFinalCommand("");
      }
    };
    fetchDefaults();
  }, []);

  // --- Handlers ---

  // Handler for generating ideas (called from IdeationTab)
  const handleGenerateIdeas = async () => {
    if (!ideationInputText.trim()) {
      setIdeationError("Error: 'Input Text for Ideation' is required.");
      return;
    }
    setIsIdeating(true);
    setGeneratedIdeas([]);
    setIdeationError("");

    const requestBody = {
      input_text: ideationInputText,
      other_info: ideationOtherInfo || null,
      initial_idea: ideationInitialIdea || null,
      num_ideas: numIdeas || 3, // Added num_ideas, fallback to 3 if null/0
    };

    try {
      console.log("Calling backend /api/v1/manim/ideate with:", requestBody);
      
      const response = await fetch('http://localhost:8000/api/v1/manim/ideate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        let errorDetail = 'Failed to fetch ideas';
        try {
          const errorData = await response.json();
          errorDetail = errorData.detail || response.statusText;
        } catch (jsonError) {
          // If parsing error JSON fails, use the status text
          errorDetail = response.statusText;
        }
        throw new Error(errorDetail);
      }

      const data = await response.json();
      setGeneratedIdeas(data.generated_ideas || []);

    } catch (error) {
      console.error("Ideation API call failed:", error);
      setIdeationError(`Error generating ideas: ${error.message}`);
      setGeneratedIdeas([]);
    } finally {
      setIsIdeating(false);
    }
  };

  // Handler for selecting an idea (called from IdeationTab)
  const handleIdeaSelection = (selectedIdea) => {
    setScriptSegment(selectedIdea); // Pre-populate the generation script segment
    // Optionally clear other generation fields if needed
    // setPreviousCodeAttempt(""); 
    // setEnhancementRequest("");
    setActiveTab('Generation'); // Switch to the Generation tab
  };

  // Handler for running the main agent (passed to GenerationTab)
  const handleRunAgent = async () => {
    setIsGenerating(true);
    setGenerationOutput("");

    // Validation (can also be done within GenerationTab)
    if (!scriptSegment.trim()) {
      setGenerationOutput("Error: 'Script Segment to Animate' is required.");
      setIsGenerating(false);
      return;
    }
    if (!sceneName.trim()) {
        setGenerationOutput("Error: 'Scene Name' is required.");
        setIsGenerating(false);
        return;
    }

    const requestBody = {
      script_segment: scriptSegment,
      general_context: generalContext || null,
      previous_code_attempt: previousCodeAttempt || null,
      enhancement_request: enhancementRequest || null,
      final_command: finalCommand || null,
      scene_name: sceneName,
      save_generated_code: saveGeneratedCode,
    };

    try {
      const response = await fetch('http://localhost:8000/run/manim_agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      const result = await response.json(); // Assume backend returns the final state or relevant parts

      // Improved Output Processing (Example)
      let output = `--- Run Result (${response.status}) ---\
`;
      output += `Success: ${result.success || false}\
`;
      output += `Message: ${result.message || (response.ok ? 'Completed' : 'Failed')}\
`;

      if (result.final_artifact_path) {
        output += `Final Artifact: ${result.final_artifact_path}\
`;
      } else if (result.final_output_path) {
        output += `Final Code Output: ${result.final_output_path}\
`;
      }
      // Add history if present (consider summarizing long histories)
      if (result.final_state?.error_history?.length) {
        output += `\nError History:\n - ${result.final_state.error_history.join('\n - ')}\
`;
      }
       if (result.final_state?.evaluation_history?.length) {
        output += `\nEvaluation History:\n - ${result.final_state.evaluation_history.join('\n - ')}\
`;
      }

      setGenerationOutput(output.trim());

    } catch (error) {
      setGenerationOutput(`Network or parsing error: ${error.message}`);
      console.error("API call failed:", error);
    } finally {
      setIsGenerating(false);
    }
  };

  // --- Render --- 
  return (
    <>
      <h1>Manim Agent Launcher</h1>

      {/* Tab Navigation */}
      <div className="tab-navigation">
        <button 
          className={activeTab === 'Ideation' ? 'active' : ''} 
          onClick={() => setActiveTab('Ideation')}
        >
          1. Ideation
        </button>
        <button 
          className={activeTab === 'Generation' ? 'active' : ''} 
          onClick={() => setActiveTab('Generation')}
        >
          2. Generation & Run
        </button>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'Ideation' && (
          <IdeationTab 
            ideationInputText={ideationInputText}
            setIdeationInputText={setIdeationInputText}
            ideationOtherInfo={ideationOtherInfo}
            setIdeationOtherInfo={setIdeationOtherInfo}
            ideationInitialIdea={ideationInitialIdea}
            setIdeationInitialIdea={setIdeationInitialIdea}
            numIdeas={numIdeas}
            setNumIdeas={setNumIdeas}
            generatedIdeas={generatedIdeas}
            isIdeating={isIdeating}
            ideationError={ideationError}
            onGenerateIdeas={handleGenerateIdeas}
            onSelectIdea={handleIdeaSelection}
          />
        )}

        {activeTab === 'Generation' && (
          <GenerationTab 
            scriptSegment={scriptSegment}
            setScriptSegment={setScriptSegment}
            generalContext={generalContext}
            setGeneralContext={setGeneralContext}
            previousCodeAttempt={previousCodeAttempt}
            setPreviousCodeAttempt={setPreviousCodeAttempt}
            enhancementRequest={enhancementRequest}
            setEnhancementRequest={setEnhancementRequest}
            finalCommand={finalCommand}
            setFinalCommand={setFinalCommand}
            sceneName={sceneName}
            setSceneName={setSceneName}
            saveGeneratedCode={saveGeneratedCode}
            setSaveGeneratedCode={setSaveGeneratedCode}
            isGenerating={isGenerating}
            generationOutput={generationOutput}
            errorLoadingDefaults={errorLoadingDefaults}
            onRunAgent={handleRunAgent}          
          />
        )}
      </div>
    </>
  );
}

export default App
