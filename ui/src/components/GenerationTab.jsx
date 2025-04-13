import React from 'react';
import PropTypes from 'prop-types';

function GenerationTab({ 
  scriptSegment,
  setScriptSegment,
  generalContext,
  setGeneralContext,
  previousCodeAttempt,
  setPreviousCodeAttempt,
  enhancementRequest,
  setEnhancementRequest,
  finalCommand,
  setFinalCommand,
  sceneName,
  setSceneName,
  saveGeneratedCode,
  setSaveGeneratedCode,
  isGenerating,
  generationOutput,
  errorLoadingDefaults,
  onRunAgent 
}) {

  return (
    <div className="generation-tab">
      <h2>2. Configure Generation & Run</h2>
      <p>Refine the script segment (pre-populated if you selected an idea) and other parameters, then run the agent.</p>

      {errorLoadingDefaults && <p className="error-message">{errorLoadingDefaults}</p>}

      <label htmlFor="scriptSegment">Script Segment to Animate <span style={{color: 'red'}}>*</span></label>
      <textarea
        id="scriptSegment"
        value={scriptSegment}
        onChange={(e) => setScriptSegment(e.target.value)}
        required
        aria-required="true"
        placeholder="The specific text segment the Manim scene should visualize. (Populated from selected idea)"
      />

      <label htmlFor="generalContext">General Context</label>
      <textarea
        id="generalContext"
        value={generalContext}
        onChange={(e) => setGeneralContext(e.target.value)}
        placeholder="General instructions, style guidance, etc. (Loaded from defaults)"
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
        placeholder="The final instruction for the LLM (e.g., Generate the code...) (Loaded from defaults)"
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

      <button onClick={onRunAgent} disabled={isGenerating || !scriptSegment.trim() || !sceneName.trim()}>
        {isGenerating ? 'Running...' : 'Run Manim Agent'}
      </button>

      {generationOutput && (
        <div className="output-container">
          <h3>Agent Output:</h3>
          <pre style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>{generationOutput}</pre>
        </div>
      )}
    </div>
  );
}

GenerationTab.propTypes = {
  scriptSegment: PropTypes.string.isRequired,
  setScriptSegment: PropTypes.func.isRequired,
  generalContext: PropTypes.string.isRequired,
  setGeneralContext: PropTypes.func.isRequired,
  previousCodeAttempt: PropTypes.string.isRequired,
  setPreviousCodeAttempt: PropTypes.func.isRequired,
  enhancementRequest: PropTypes.string.isRequired,
  setEnhancementRequest: PropTypes.func.isRequired,
  finalCommand: PropTypes.string.isRequired,
  setFinalCommand: PropTypes.func.isRequired,
  sceneName: PropTypes.string.isRequired,
  setSceneName: PropTypes.func.isRequired,
  saveGeneratedCode: PropTypes.bool.isRequired,
  setSaveGeneratedCode: PropTypes.func.isRequired,
  isGenerating: PropTypes.bool.isRequired,
  generationOutput: PropTypes.string.isRequired,
  errorLoadingDefaults: PropTypes.string.isRequired,
  onRunAgent: PropTypes.func.isRequired,
};

export default GenerationTab; 