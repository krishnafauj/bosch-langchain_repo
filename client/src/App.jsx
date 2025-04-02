import { useState } from "react";
import { motion } from "framer-motion";
import { Send, Smile, Paperclip, MoreVertical, Search } from "lucide-react";
import imageSrc from "./image.png";

export default function App() {
  const toolLinks = {
    "GWB 12V-10": "https://www.bosch-professional.com/gb/en/products/gwb-12v-10-0601390909",
    "GBH 12-52 DV": "https://www.bosch-pt.co.in/in/en/products/gbh-12-52-dv-0611266000",
    "GBH 2-26 DFR": "https://www.bosch-professional.com/kw/en/products/gbh-2-26-dfr-06112547P0",
    "GBH 2-28": "https://www.bosch-pt.co.in/in/en/products/gbh-2-28-dv-06112671F0",
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
    "GBH 18V-26F": "https://www.bosch-professional.com/gb/en/products/gbh-18v-26f-0611910000",
    "GSB 12V-15": "https://www.bosch-professional.com/gb/en/products/gsb-12v-15-06019B6901",
    "GSR 12V-15": "https://www.bosch-pt.co.in/in/en/products/gsr-12v-15-fc-06019F60F0",
    "GSR 18V-60 C": "https://www.bosch-professional.com/gb/en/products/gsr-18v-60-c-06019G1102",
    "GSR 18V-110 C": "https://www.bosch-professional.com/gb/en/products/gsr-18v-110-c-06019G0109",
    "GSR 18V-20": "https://www.bosch-professional.com/gb/en/products/gbh-18v-20-0611911000",
    "GSR 18V-50": "https://www.bosch-pt.co.in/in/en/products/gsr-18v-50-06019H5082",
    "GSB 18V-LI": "https://www.bosch-pt.co.in/in/en/products/gsr-18v-50-06019H5082",

  }
  const [messages, setMessages] = useState([
    { id: 2, text: "Hi! How are you?", sender: "me", time: "09:42" },
    { id: 1, text: "I am Bosch Tool Finder! 🛠️ I am here to assist you in finding the right tools. How can I help you today?", sender: "them", time: "09:42" },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  
  const sendMessage = async () => {
    if (!input.trim()) return;
    const newMessages = [
      ...messages,
      {
        text: input,
        sender: "me",
        time: new Date().toLocaleTimeString("en-US", {
          hour: "2-digit",
          minute: "2-digit",
        }),
      },
    ];
    setMessages(newMessages);
    setInput("");
    setLoading(true);
  
    try {
      const response = await fetch(
        "https://bosch-langchain-repo-4.onrender.com/ask",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: input }),
        }
      );
      const data = await response.json();
      let botResponse = data.answer.replace(/\n/g, "<br />");
  
      // Replace tool keywords with hyperlinks
      for (const tool in toolLinks) {
        const toolRegex = new RegExp(`\\b${tool}\\b`, "g"); // Match whole word
        botResponse = botResponse.replace(
          toolRegex,
          `<a href="${toolLinks[tool]}" target="_blank" class="text-blue-400 underline">${tool}</a>`
        );
      }
  
      setMessages([
        ...newMessages,
        {
          text: botResponse,
          sender: "them",
          time: new Date().toLocaleTimeString("en-US", {
            hour: "2-digit",
            minute: "2-digit",
          }),
        },
      ]);
    } catch (error) {
      console.error("Error fetching response:", error);
    }
    setLoading(false);
  };
  

  return (
    <div className="flex h-screen min-w-screen bg-gray-500">
      <img src="" alt="" />
      <div className="flex flex-col flex-1 bg-gray-600 shadow-xl  mb-4 overflow-hidden">
        <div className="flex items-center justify-between  border-b">
          <div className="flex items-center ">
            <img src={imageSrc} alt="Logo" className="h-10 w-full " />
          
          </div>

        </div>

        <div className="flex-1 flex flex-col  items-center overflow-y-auto p-6 space-y-4 ">
          {messages.map((msg, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`w-fit max-w-lg px-4 py-2 rounded-2xl ${msg.sender === "me"
                  ? "bg-blue-500 text-white rounded-br-none self-end"
                  : "bg-gray-100 text-gray-800 rounded-bl-none self-start"
                }`}
              dangerouslySetInnerHTML={{ __html: msg.text }}
            />
          ))}
          {loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="bg-gray-700 px-4 py-2 rounded-lg text-white"
            >
              ...
            </motion.div>
          )}
        </div>


        <div className="px-6 py-4 border-t bg-black">
          <div className="flex items-center  space-x-4 ">
            <button className="p-2 hover:bg-gray-100 rounded-full"><Paperclip className="w-5 h-5 text-gray-500" /></button>
            <div className="flex-1 flex items-center space-x-4 bg-gray-600 rounded-full px-4 py-2">
              <input
                type="text"
                placeholder="Type a message..."
                className="flex-1 bg-transparent outline-none"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && sendMessage()}
              />
              <button className="p-2 hover:bg-gray-200 rounded-full"><Smile className="w-5 h-5 text-gray-500" /></button>
            </div>
            <button onClick={sendMessage} className="p-3 bg-blue-500 hover:bg-blue-600 rounded-full text-white">
              <Send className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}