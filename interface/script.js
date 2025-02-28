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

        // Simulate bot response
        setTimeout(() => {
            const botResponse = document.createElement("div");
            botResponse.classList.add("chat-bubble", "bot-message");
            botResponse.textContent = "This is a bot response!";
            chatHistory.appendChild(botResponse);

            // Scroll to latest message
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }, 1000);

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
});
