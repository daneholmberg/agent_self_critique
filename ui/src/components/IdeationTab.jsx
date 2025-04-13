import React from 'react';
import PropTypes from 'prop-types';

function IdeationTab({ 
  ideationInputText,
  setIdeationInputText,
  ideationOtherInfo,
  setIdeationOtherInfo,
  ideationInitialIdea,
  setIdeationInitialIdea,
  numIdeas,
  setNumIdeas,
  generatedIdeas,
  isIdeating,
  ideationError,
  onGenerateIdeas,
  onSelectIdea
}) {

  return (
    <div className="ideation-tab">
      <h2>1. Generate Scene Ideas</h2>
      <p>Describe the core concept you want to animate. Optionally provide an initial idea or direction.</p>

      <label htmlFor="ideationOtherInfo">Other Relevant Information (Optional)</label>
      <textarea
        id="ideationOtherInfo"
        value={ideationOtherInfo}
        onChange={(e) => setIdeationOtherInfo(e.target.value)}
        placeholder="e.g., The target audience is high school students. Keep the visuals simple and clear."
      />

      <label htmlFor="ideationInputText">Input Text for Ideation <span style={{color: 'red'}}>*</span></label>
      <textarea
        id="ideationInputText"
        value={ideationInputText}
        onChange={(e) => setIdeationInputText(e.target.value)}
        required
        aria-required="true"
        placeholder="e.g., Explain the Pythagorean theorem visually."
      />

      <label htmlFor="ideationInitialIdea">Initial Idea (Optional)</label>
      <textarea
        id="ideationInitialIdea"
        value={ideationInitialIdea}
        onChange={(e) => setIdeationInitialIdea(e.target.value)}
        placeholder="e.g., Maybe show a right-angled triangle with squares on each side?"
      />

      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '15px' }}>
        <button onClick={onGenerateIdeas} disabled={isIdeating || !ideationInputText.trim() || numIdeas <= 0}>
          {isIdeating ? 'Generating...' : 'Generate Ideas'}
        </button>
        <label htmlFor="numIdeas" style={{ marginBottom: '0' }}>Number of ideas:</label>
        <input 
          type="number" 
          id="numIdeas" 
          value={numIdeas} 
          onChange={(e) => setNumIdeas(parseInt(e.target.value, 10) || 0)} 
          min="1" 
          max="10"
          style={{ width: '60px' }} 
        />
      </div>

      {ideationError && <p className="error-message">{ideationError}</p>}

      {generatedIdeas.length > 0 && (
        <div className="ideas-container">
          <h3>Select an Idea (or proceed to Generation tab)</h3>
          <ul>
            {generatedIdeas.map((idea, index) => (
              <li key={index} onClick={() => onSelectIdea(idea)} className="idea-item">
                {idea}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

IdeationTab.propTypes = {
  ideationInputText: PropTypes.string.isRequired,
  setIdeationInputText: PropTypes.func.isRequired,
  ideationOtherInfo: PropTypes.string.isRequired,
  setIdeationOtherInfo: PropTypes.func.isRequired,
  ideationInitialIdea: PropTypes.string.isRequired,
  setIdeationInitialIdea: PropTypes.func.isRequired,
  numIdeas: PropTypes.number.isRequired,
  setNumIdeas: PropTypes.func.isRequired,
  generatedIdeas: PropTypes.arrayOf(PropTypes.string).isRequired,
  isIdeating: PropTypes.bool.isRequired,
  ideationError: PropTypes.string.isRequired,
  onGenerateIdeas: PropTypes.func.isRequired,
  onSelectIdea: PropTypes.func.isRequired,
};

export default IdeationTab; 