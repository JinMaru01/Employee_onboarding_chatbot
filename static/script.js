document.addEventListener("DOMContentLoaded", function () {
    const textarea = document.getElementById("chat-input");

    textarea.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = this.scrollHeight + "px";
    });
});

document.addEventListener("DOMContentLoaded", function () {
    const textarea = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const chatHistory = document.getElementById("chat-history");

    function handleUserInput() {
        const message = textarea.value.trim();
        if (message === "") return;

        // Show chat history panel on first message
        if (chatHistory.children.length === 0) {
            chatHistory.style.display = "flex";
        }

        // Create user message
        const userMessage = document.createElement("div");
        userMessage.classList.add("chat-bubble", "user-message");
        userMessage.textContent = message;
        chatHistory.appendChild(userMessage);

        fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: message })
        })
        .then(response => response.json())
        .then(data => {
            const botResponse = document.createElement("div");
            botResponse.classList.add("chat-bubble", "bot-message");

            const intent = data.predicted_intent;
            const response_time = data.prediction_time;
            const generated_response = data.generated_response;
            const confidence = (data.confidence * 100).toFixed(2);

            botResponse.textContent = `Intent: ${intent} (Confidence: ${confidence}%), with ${response_time}s, \n ${generated_response}`;
            // botResponse.textContent = `${generated_response}`;
            chatHistory.appendChild(botResponse);
            chatHistory.scrollTop = chatHistory.scrollHeight;
        })
        .catch(error => {
            const errorMsg = document.createElement("div");
            errorMsg.classList.add("chat-bubble", "bot-message", "error");
            errorMsg.textContent = "Error: Unable to fetch response.";
            chatHistory.appendChild(errorMsg);
            chatHistory.scrollTop = chatHistory.scrollHeight;
        });

        // Scroll to latest message
        chatHistory.scrollTop = chatHistory.scrollHeight;

        // Clear input field
        textarea.value = "";
        textarea.style.height = "auto";
    }

    // Auto-expand input
    textarea.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = this.scrollHeight + "px";
    });

    // Send message on button click
    sendBtn.addEventListener("click", handleUserInput);

    // Send message on Enter (Shift+Enter for a new line)
    textarea.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            handleUserInput();
        }
    });

    // Hide chat history initially
    chatHistory.style.display = "none";
    
    const clearBtn = document.getElementById("clear-btn");
    
    clearBtn.addEventListener("click", function () {
        chatHistory.innerHTML = "";
        chatHistory.style.display = "none";
    });
});

