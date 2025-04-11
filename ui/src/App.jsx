import { useState } from 'react'
// import reactLogo from './assets/react.svg'
// import viteLogo from '/vite.svg'
import './App.css'

function App() {
  // State variables
  const [scriptSegment, setScriptSegment] = useState("");
  const [scriptContext, setScriptContext] = useState("");
  const [metadata, setMetadata] = useState("");
  const [saveGeneratedCode, setSaveGeneratedCode] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [responseOutput, setResponseOutput] = useState("");

  // API call handler
  const handleRunAgent = async () => {
    setIsLoading(true);
    setResponseOutput(""); // Clear previous output

    // Basic validation
    if (!scriptSegment.trim()) {
      setResponseOutput("Error: 'Script Segment to Animate' is required.");
      setIsLoading(false);
      return;
    }

    const requestBody = {
      script_segment: scriptSegment,
      script_context: scriptContext || null, // Send null if empty
      metadata: metadata || null,          // Send null if empty
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
        let output = `Status: ${result.success ? 'Success' : 'Failure'}\n\n`;
        if (result.stdout) {
          output += `--- Stdout ---\n${result.stdout}\n\n`;
        }
        if (result.stderr) {
          output += `--- Stderr ---\n${result.stderr}\n`;
        }
        setResponseOutput(output.trim());
      } else {
        // Handle HTTP errors (e.g., 4xx, 5xx)
        setResponseOutput(`Error: ${response.status} ${response.statusText}\n${result.detail || 'Unknown error'}`);
      }
    } catch (error) {
      // Handle network errors or JSON parsing errors
      setResponseOutput(`Network or parsing error: ${error.message}`);
      console.error("API call failed:", error);
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <>
      {/* Remove boilerplate div with logos */}
      {/* <div> ... logos ... </div> */}
      <h1>Manim Agent Launcher</h1>

      {/* Script Segment Input */}
      <label htmlFor="scriptSegment">Script Segment to Animate <span style={{color: 'red'}}>*</span></label>
      <textarea
        id="scriptSegment"
        value={scriptSegment}
        onChange={(e) => setScriptSegment(e.target.value)}
        required
        aria-required="true"
      />

      {/* Script Context Input */}
      <label htmlFor="scriptContext">Optional Full Script Context</label>
      <textarea
        id="scriptContext"
        value={scriptContext}
        onChange={(e) => setScriptContext(e.target.value)}
      />

      {/* Metadata Input */}
      <label htmlFor="metadata">Optional Metadata (JSON string)</label>
      <input
        type="text"
        id="metadata"
        value={metadata}
        onChange={(e) => setMetadata(e.target.value)}
        placeholder='e.g., {"scene_name": "MyScene"}'
      />

      {/* New Toggle Switch for Saving Code */}
      <div className="toggle-switch">
        <input
          type="checkbox"
          id="saveCodeToggle"
          checked={saveGeneratedCode}
          onChange={(e) => setSaveGeneratedCode(e.target.checked)}
        />
        <label htmlFor="saveCodeToggle">Save Generated Code Per Iteration</label>
      </div>

      {/* Run Button */}
      <button onClick={handleRunAgent} disabled={isLoading}>
        {isLoading ? 'Running...' : 'Run Manim Agent'}
      </button>

      {/* Response Output */}
      {responseOutput && (
        <div>
          <h2>Output:</h2>
          <pre>{responseOutput}</pre>
        </div>
      )}

      {/* Remove boilerplate card and paragraph */}
      {/* <div className="card"> ... </div> */}
      {/* <p className="read-the-docs"> ... </p> */}
    </>
  )
}

export default App
