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

        fetch("/api/respond", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: message })
        })
        .then(response => response.json())
        .then(data => {
            const { predicted_intent, confidence, response_time, response, entities } = data;

            // Show intent and confidence
            const intentBubble = document.createElement("div");
            intentBubble.classList.add("chat-bubble", "bot-message");
            // intentBubble.textContent = `Intent: ${predicted_intent} (Confidence: ${(confidence * 100).toFixed(2)}%), took ${response_time}s`;
            // intentBubble.textContent = `(Confidence: ${(confidence ).toFixed(2)}%) ${response}`
            // chatHistory.appendChild(intentBubble);

            // Show final response
            if (response) {
                const responseBubble = document.createElement("div");
                responseBubble.classList.add("chat-bubble", "bot-message");
                responseBubble.textContent = response;
                chatHistory.appendChild(responseBubble);
            }

            // Show extracted entities (if any)
            // if (entities && Object.keys(entities).length > 0) {
            //     const entityBubble = document.createElement("div");
            //     entityBubble.classList.add("chat-bubble", "bot-message");

            //     const entityText = Object.entries(entities)
            //         .map(([type, value]) => `${value} [${type}]`)
            //         .join(", ");

            //     entityBubble.textContent = `Entities: ${entityText}`;
            //     chatHistory.appendChild(entityBubble);
            // }

            // Scroll to latest message
            chatHistory.scrollTop = chatHistory.scrollHeight;
        })
        .catch(error => {
            const errorBubble = document.createElement("div");
            errorBubble.classList.add("chat-bubble", "bot-message", "error");
            errorBubble.textContent = "Error: Unable to get response from chatbot.";
            chatHistory.appendChild(errorBubble);
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

    // Send message on Enter (Shift+Enter for newline)
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
