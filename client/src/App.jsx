import { useState } from "react";
import { motion } from "framer-motion";
import { Send } from "lucide-react";
import imageSrc from "./image.png";

export default function Chatbot() {
  const toolLinks = {
    "GWB 12V-10": "https://www.bosch-professional.com/gb/en/products/gwb-12v-10-0601390909",
    "GWB 10 RE": "https://www.bosch-professional.com/lb/en/products/gwb-10-re-0601132703",
    "GBH 2-26 DRE": "https://www.bosch-professional.com/gb/en/products/gbh-2-26-06112A3060?queryFromSuggest=true&userInput=GBH+2-26+DRE",
    "GBH 18V-40 C": "https://www.bosch-professional.com/gb/en/products/gbh-18v-40-c-0611917100?queryFromSuggest=true&userInput=GBH+18V-40+C",
    "GBH 5-40 D": "https://www.bosch-professional.com/gb/en/products/gbh-5-40-d-0611269060?queryFromSuggest=true&userInput=GBH+5-40+D",
    "GBH 18V-45 C": "https://www.bosch-professional.com/gb/en/products/gbh-18v-45-c-0611913000?queryFromSuggest=true&userInput=GBH+18V-45+C",
    "GBH 4-32 DFR": "https://www.bosch-professional.com/gb/en/products/gbh-4-32-dfr-0611332161",
    "GTB18V-45": "https://www.bosch-professional.com/gb/en/products/gtb-18v-45-06019K7000?queryFromSuggest=true&userInput=GTB18V-45",
    "GTB12V-11": "https://www.bosch-professional.com/gb/en/products/gtb-12v-11-06019E4002?queryFromSuggest=true&userInput=GTB12V-11",
    "GTB185-LI w/ GMA55": "https://www.bosch-professional.com/gb/en/products/gho-18v-li-06015A0300?queryFromSuggest=true&userInput=GTB185-LI+w%2F+GMA55",
    "GSR 18V-90 FC": "https://www.bosch-professional.com/gb/en/products/gsr-18v-90-fc-06019K6202",
    "GSR 18V-90 C": "https://www.bosch-professional.com/gb/en/products/gsr-18v-90-c-06019K6000",
    "GBM 13-2 RE": "https://www.bosch-professional.com/gb/en/products/gbm-13-2-re-06011B2060",
    "GSR 18V-55": "https://www.bosch-professional.com/gb/en/products/gsr-18v-55-06019H5202",
    "GSB 18V-45": "https://www.bosch-professional.com/gb/en/products/gsb-18v-45-06019K3300",
    "GSB 21-2 RE": "https://www.bosch-professional.com/gb/en/products/gsb-21-2-re-060119C560",
    "GSB 18V-21": "https://www.bosch-professional.com/gb/en/products/gsb-18v-21-06019H1108",
    "GSB 18V-55": "https://www.bosch-professional.com/gb/en/products/gsb-18v-55-06019H5302",
    "GSB 18V-28": "https://www.bosch-professional.com/gb/en/products/gsb-18v-28-06019H4000",
    "GSB 162-2 RE": "https://www.bosch-professional.com/gb/en/products/gsb-162-2-re-060118B060",
    "GSB 12V-15": "https://www.bosch-professional.com/gb/en/products/gsb-12v-15-06019B6901",
    "GSR 12V-15": "https://www.bosch-pt.co.in/in/en/products/gsr-12v-15-fc-06019F60F0",
    "GSR 18V-60 C": "https://www.bosch-professional.com/gb/en/products/gsr-18v-60-c-06019G1102",
    "GSR 18V-110 C": "https://www.bosch-professional.com/gb/en/products/gsr-18v-110-c-06019G0109",
    "GSB 18V-110 C": "https://www.bosch-professional.com/sa/en/products/gsb-18v-110-c-06019G030A"
  };
  
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const newMessages = [...messages, { text: input, sender: "user" }];
    setMessages(newMessages);
    setInput("");
    setLoading(true);
    
    try {
      const response = await fetch("https://bosch-langchain-repo-2.onrender.com/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: input })
      });
      const data = await response.json();
  
      let botResponse = data.answer.replace(/\n/g, "<br />");
      let links = "";
  
      // Check if any tool name exists in the response and collect their links
      for (const tool in toolLinks) {
        if (botResponse.includes(tool)) 
        {
          links += `<br /><a href="${toolLinks[tool]}" target="_blank" class="text-blue-400 underline">More about ${tool}</a>`;
        }
        
      }
  
      if (links) {
        botResponse += `<br /><br />Related Tools:${links}`;
      }
  
      setMessages([...newMessages, { text: botResponse, sender: "bot" }]);
    } catch (error) {
      console.error("Error fetching response:", error);
    }
    setLoading(false);
  };
  

  return (<div className="flex flex-col  min-w-screen items-center gap-5 h-screen bg-gray-900 text-white ">
    <img src={imageSrc} alt="Description" />

      <div className="flex flex-col w-full pl-50 max-w-5xl mb-5 pb-24 m-2 space-y-5 flex-grow overflow-auto">
        {messages.map((msg, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`max-w-lg px-4 py-2 rounded-lg ${
              msg.sender === "user" ? "ml-auto bg-blue-500" : "mr-auto bg-gray-700"
            }`}
            style={{ marginRight: msg.sender === "user" ? 200 : 0, marginLeft: msg.sender === "bot" ? 200 : 0 }}
            dangerouslySetInnerHTML={{ __html: msg.text }}
          />
        ))}
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mr-auto bg-gray-700 px-4 py-2 rounded-lg"
            style={{ marginLeft: 200 }}
          >
            ...
          </motion.div>
        )}
      </div>
      <div className="flex items-center w-full max-w-lg p-4  border-t border-gray-700 fixed bottom-0 bg-gray-900">
        <input
          className="flex-1 px-4 py-2 rounded-lg bg-gray-800 border border-gray-600 text-white focus:outline-none"
          placeholder="Ask something..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && sendMessage()}
        />
        <button
          className="ml-4 p-2 bg-blue-500 rounded-lg hover:bg-blue-600"
          onClick={sendMessage}
        >
          <Send className="w-6 h-6 text-white" />
        </button>
      </div>
    </div>
  );
}
