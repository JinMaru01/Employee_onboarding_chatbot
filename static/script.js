document.addEventListener("DOMContentLoaded", function () {
    const textarea = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const chatHistory = document.getElementById("chat-history");
    const faqSuggestions = document.getElementById("faq-suggestions");
    const faqList = document.getElementById("faq-list");

    const mostAskedQuestions = [
        "What is the company vision?",
        "Tell me about our core values",
        "What types of contracts are available?",
        "How do I apply for leave?",
        "What are the unacceptable actions?",
        "Can you explain the disciplinary procedure?"
    ];

    // Always show on load
    faqSuggestions.style.display = "block";
    faqList.innerHTML = ""; // clear old content
    mostAskedQuestions.forEach(question => {
        const li = document.createElement("li");
        li.className = "faq-item";
        li.innerHTML = `<button class="btn btn-link p-0 text-left faq-btn">${question}</button>`;
        faqList.appendChild(li);
    });

    function handleUserInput() {
        const message = textarea.value.trim();
        if (message === "") return;

        // Hide FAQ after interaction
        faqSuggestions.style.display = "none";

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
            const { response } = data;

            if (response) {
                // Typing animation placeholder
                const typingBubble = document.createElement("div");
                typingBubble.classList.add("chat-bubble", "bot-message");
                typingBubble.textContent = "Typing...";
                chatHistory.appendChild(typingBubble);
                chatHistory.scrollTop = chatHistory.scrollHeight;

                // Simulate delay before showing actual message
                setTimeout(() => {
                    typingBubble.remove(); // remove "Typing..."

                    const responseBubble = document.createElement("div");
                    responseBubble.classList.add("chat-bubble", "bot-message");
                    responseBubble.textContent = response;
                    chatHistory.appendChild(responseBubble);
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                }, 800); // adjust delay in ms
            }
        })

    setTimeout(() => {
        typingBubble.remove();

        const responseBubble = document.createElement("div");
        responseBubble.classList.add("chat-bubble", "bot-message");
        chatHistory.appendChild(responseBubble);

        let i = 0;
        function typeChar() {
            if (i < response.length) {
                responseBubble.textContent += response.charAt(i);
                i++;
                setTimeout(typeChar, 20); // adjust typing speed
            }
        }
        typeChar();
    }, 800);



        textarea.value = "";
        textarea.style.height = "auto";
    }

    // FAQ button click
    faqList.addEventListener("click", function (e) {
        if (e.target.classList.contains("faq-btn")) {
            textarea.value = e.target.textContent;
            handleUserInput();
        }
    });

    // Button and Enter
    sendBtn.addEventListener("click", handleUserInput);
    textarea.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            handleUserInput();
        }
    });

    let historyToDelete = null; // store the item to delete when confirmed

    // Show modal function
    function showConfirmModal() {
        document.getElementById('confirm-modal').style.display = 'flex';
    }

    // Hide modal function
    function hideConfirmModal() {
        document.getElementById('confirm-modal').style.display = 'none';
    }

      // Attach click listeners to delete buttons
    document.querySelectorAll('.delete-history-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
        e.preventDefault();
        historyToDelete = btn.closest('.sidebar-link');
        if (historyToDelete) {
            showConfirmModal();
        }
        });
    });

    // Confirm Yes button
    document.getElementById('confirm-yes').addEventListener('click', () => {
        if (historyToDelete) {
        historyToDelete.remove();
        historyToDelete = null;
        // TODO: Add your backend delete request here if needed
        }
        hideConfirmModal();
    });

    // Confirm No button
    document.getElementById('confirm-no').addEventListener('click', () => {
        historyToDelete = null;
        hideConfirmModal();
    });

    // Optional: close modal on clicking outside modal-content
    document.getElementById('confirm-modal').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) {
        hideConfirmModal();
        historyToDelete = null;
        }
    });
});

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    sidebar.classList.toggle('collapsed');
}

function addToSidebar(message) {
    const li = document.createElement("li");
    li.className = "list-group-item small";
    li.textContent = message;
    document.getElementById("chat-history-sidebar").appendChild(li);
}
