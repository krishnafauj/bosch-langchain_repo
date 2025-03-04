import React, { useState } from 'react';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSendQuestion = async () => {
    if (!question.trim()) return;

    setIsLoading(true);

    try {
      const response = await fetch('https://bosch-langchain-repo-2.onrender.com/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });

      const data = await response.json();
      setResponse(data.answer.replace(/\n/g, ' ')); // Replace \n with spaces
    } catch (error) {
      console.error('Error:', error);
      setResponse('Failed to fetch response.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-box">
        <div className="response-box">
          {response && <div className="response">{response}</div>}
        </div>
        <div className="question-box">
          {question && <div className="question">{question}</div>}
        </div>
      </div>
      <div className="input-container">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Type your question..."
        />
        <button onClick={handleSendQuestion} disabled={isLoading}>
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  );
}

export default App;