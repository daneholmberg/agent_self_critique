import { useState, useEffect } from 'react'
// import reactLogo from './assets/react.svg'
// import viteLogo from '/vite.svg'
import './App.css'

function App() {
  // State variables
  const [scriptSegment, setScriptSegment] = useState("");
  const [generalContext, setGeneralContext] = useState("");
  const [previousCodeAttempt, setPreviousCodeAttempt] = useState("");
  const [enhancementRequest, setEnhancementRequest] = useState("");
  const [finalCommand, setFinalCommand] = useState("");
  const [sceneName, setSceneName] = useState("");
  const [saveGeneratedCode, setSaveGeneratedCode] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [responseOutput, setResponseOutput] = useState("");
  const [errorLoadingDefaults, setErrorLoadingDefaults] = useState("");

  // Fetch default configuration on component mount
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
        setGeneralContext("");
        setFinalCommand("");
      }
    };
    fetchDefaults();
  }, []);

  // API call handler
  const handleRunAgent = async () => {
    setIsLoading(true);
    setResponseOutput("");

    if (!scriptSegment.trim()) {
      setResponseOutput("Error: 'Script Segment to Animate' is required.");
      setIsLoading(false);
      return;
    }
    if (!sceneName.trim()) {
        setResponseOutput("Error: 'Scene Name' is required.");
        setIsLoading(false);
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
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      const result = await response.json();

      if (response.ok) {
        let output = `--- Run Result ---`;
        if (result.error_history && result.error_history.length > 0) {
            output += `\\n\\nErrors Encountered:\\n${result.error_history.join('\\n')}`;
        }
        if (result.evaluation_history && result.evaluation_history.length > 0) {
            output += `\\n\\nEvaluation History:\\n${result.evaluation_history.join('\\n')}`;
        }
        if (result.validated_artifact_path) {
             output += `\\n\\nValidated Artifact: ${result.validated_artifact_path}`;
        } else if (result.generated_output && !result.validation_error) {
             output += `\\n\\nGenerated Code (Validation Pending/Failed):\n\`\`\`python\n${result.generated_output}\n\`\`\``;
        } else if (result.validation_error) {
            output += `\\n\\nValidation Error: ${result.validation_error}`;
        } else if (result.generated_output) {
            output += `\\n\\nGenerated Output:\\n${result.generated_output}`;
        } else if (!result.error_history || result.error_history.length === 0) {
            output += "\\n\\nExecution initiated, check server logs/output directory for details.";
        }

        setResponseOutput(output.trim());

      } else {
        setResponseOutput(`Error: ${response.status} ${response.statusText}\\n${result.detail || 'Unknown error'}`);
      }
    } catch (error) {
      setResponseOutput(`Network or parsing error: ${error.message}`);
      console.error("API call failed:", error);
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <>
      <h1>Manim Agent Launcher</h1>

      {errorLoadingDefaults && <p className="error-message">{errorLoadingDefaults}</p>}

      <label htmlFor="generalContext">General Context</label>
      <textarea
        id="generalContext"
        value={generalContext}
        onChange={(e) => setGeneralContext(e.target.value)}
        placeholder="General instructions, style guidance, etc."
      />

      <label htmlFor="scriptSegment">Script Segment to Animate <span style={{color: 'red'}}>*</span></label>
      <textarea
        id="scriptSegment"
        value={scriptSegment}
        onChange={(e) => setScriptSegment(e.target.value)}
        required
        aria-required="true"
        placeholder="The specific text segment the Manim scene should visualize."
      />

      <label htmlFor="previousCodeAttempt">Previous Code Attempt (Optional)</label>
      <textarea
        id="previousCodeAttempt"
        value={previousCodeAttempt}
        onChange={(e) => setPreviousCodeAttempt(e.target.value)}
        placeholder="Paste the previous code here if you want to enhance it."
      />

      <label htmlFor="enhancementRequest">What We Want Enhanced (Optional)</label>
      <textarea
        id="enhancementRequest"
        value={enhancementRequest}
        onChange={(e) => setEnhancementRequest(e.target.value)}
        placeholder="Describe the specific changes or improvements needed for the previous code."
      />

      <label htmlFor="sceneName">Scene Name <span style={{color: 'red'}}>*</span></label>
      <input
        type="text"
        id="sceneName"
        value={sceneName}
        onChange={(e) => setSceneName(e.target.value)}
        required
        aria-required="true"
        placeholder="A short, descriptive name (used for filenames)"
      />

      <label htmlFor="finalCommand">Final Command</label>
      <textarea
        id="finalCommand"
        value={finalCommand}
        onChange={(e) => setFinalCommand(e.target.value)}
        placeholder="The final instruction for the LLM (e.g., Generate the code...)"
      />

      <div className="toggle-switch">
        <input
          type="checkbox"
          id="saveCodeToggle"
          checked={saveGeneratedCode}
          onChange={(e) => setSaveGeneratedCode(e.target.checked)}
        />
        <label htmlFor="saveCodeToggle">Save Generated Code Per Iteration</label>
      </div>

      <button onClick={handleRunAgent} disabled={isLoading}>
        {isLoading ? 'Running...' : 'Run Manim Agent'}
      </button>

      {responseOutput && (
        <div>
          <h2>Output:</h2>
          <pre style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>{responseOutput}</pre>
        </div>
      )}
    </>
  )
}

export default App
