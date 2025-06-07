document.addEventListener("DOMContentLoaded", function () {
    // DOM Elements
    const textarea = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const chatHistory = document.getElementById("chat-history");
    const faqSuggestions = document.getElementById("faq-suggestions");
    const faqList = document.getElementById("faq-list");
    const sidebar = document.getElementById("sidebar");

    const mostAskedQuestions = [
        "What is the company vision?",
        "Tell me about our core values",
        "What types of contracts are available?",
        "How do I apply for leave?",
        "What are the unacceptable actions?",
        "Can you explain the disciplinary procedure?"
    ];

    let historyToDelete = null;
    let inputDebounce = false;

    function initializeFAQ() {
        if (faqSuggestions && faqList) {
            faqSuggestions.style.display = "block";
            faqList.innerHTML = "";
            mostAskedQuestions.forEach(question => {
                const li = document.createElement("li");
                li.className = "faq-item";
                li.innerHTML = `<button class="btn btn-link p-0 text-left faq-btn">${question}</button>`;
                faqList.appendChild(li);
            });
        }
    }

    function handleUserInput() {
        if (inputDebounce) return;
        inputDebounce = true;
        setTimeout(() => inputDebounce = false, 500);

        if (!textarea) return;
        const message = textarea.value.trim();
        if (message === "") return;

        if (faqSuggestions) faqSuggestions.style.display = "none";

        addMessageToChat('user', message);
        addToSidebar(message);
        sendToAPI(message);

        textarea.value = "";
        textarea.style.height = "auto";

        if (window.innerWidth <= 768) {
            textarea.scrollIntoView({ behavior: "smooth", block: "end" });
        }
    }

    function sendToAPI(message) {
        if (!chatHistory) return;

        const typingBubble = document.createElement("div");
        typingBubble.classList.add("chat-bubble", "bot-message");
        typingBubble.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Bot is typing...`;
        chatHistory.appendChild(typingBubble);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        fetch("/api/respond", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: message })
        })
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.json();
        })
        .then(data => {
            const { response } = data;
            if (response) {
                typingBubble.remove();
                addBotResponseWithTyping(response);
            }
        })
        .catch(error => {
            console.error('API Error:', error);
            typingBubble.textContent = "Sorry, there was an error processing your request.";
        });
    }

    function addBotResponseWithTyping(response) {
        if (!chatHistory) return;

        const responseBubble = document.createElement("div");
        responseBubble.classList.add("chat-bubble", "bot-message");
        chatHistory.appendChild(responseBubble);

        let i = 0;
        function typeChar() {
            if (i < response.length) {
                responseBubble.textContent += response.charAt(i);
                i++;
                chatHistory.scrollTop = chatHistory.scrollHeight;
                setTimeout(typeChar, 20);
            }
        }
        typeChar();
    }

    function addMessageToChat(sender, message) {
        if (!chatHistory) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-bubble ${sender === 'user' ? 'user-message' : 'bot-message'}`;
        messageDiv.textContent = message;
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function setupTextareaResize() {
        if (!textarea) return;
        textarea.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
    }

    function setupEventListeners() {
        if (faqList) {
            faqList.addEventListener("click", function (e) {
                if (e.target.classList.contains("faq-btn")) {
                    textarea.value = e.target.textContent;
                    handleUserInput();
                }
            });
        }

        if (sendBtn) {
            sendBtn.addEventListener("click", handleUserInput);
        }

        if (textarea) {
            textarea.addEventListener("keydown", function (event) {
                if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    handleUserInput();
                }
            });
        }

        document.addEventListener('click', function (e) {
            if (e.target.closest('.faq-link')) {
                e.preventDefault();
                const question = e.target.closest('.faq-link').textContent.trim();
                textarea.value = question;
                handleUserInput();
            }
        });

        document.addEventListener('click', function (e) {
            if (e.target.closest('.nav-link') && !e.target.closest('.delete-btn')) {
                e.preventDefault();
                document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
                e.target.closest('.nav-link').classList.add('active');
                const navText = e.target.closest('.nav-link').querySelector('.nav-text');
                if (navText) loadChatHistory(navText.textContent);
            }
        });
    }

    function showConfirmModal() {
        const modal = document.getElementById('confirm-modal');
        if (modal) modal.style.display = 'flex';
    }

    function hideConfirmModal() {
        const modal = document.getElementById('confirm-modal');
        if (modal) modal.style.display = 'none';
    }

    function setupDeleteConfirmation() {
        document.querySelectorAll('.delete-history-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                historyToDelete = btn.closest('.sidebar-link');
                if (historyToDelete) showConfirmModal();
            });
        });

        const confirmYes = document.getElementById('confirm-yes');
        if (confirmYes) {
            confirmYes.addEventListener('click', () => {
                if (historyToDelete) historyToDelete.remove();
                historyToDelete = null;
                hideConfirmModal();
            });
        }

        const confirmNo = document.getElementById('confirm-no');
        if (confirmNo) {
            confirmNo.addEventListener('click', () => {
                historyToDelete = null;
                hideConfirmModal();
            });
        }

        const confirmModal = document.getElementById('confirm-modal');
        if (confirmModal) {
            confirmModal.addEventListener('click', (e) => {
                if (e.target === e.currentTarget) {
                    hideConfirmModal();
                    historyToDelete = null;
                }
            });
        }

        const confirmDelete = document.getElementById('confirmDelete');
        if (confirmDelete) {
            confirmDelete.addEventListener('click', function () {
                const activeNavLink = document.querySelector('.nav-link.active');
                if (activeNavLink) {
                    const navItem = activeNavLink.closest('.nav-item');
                    if (navItem) navItem.remove();
                    const remainingItems = document.querySelectorAll('.nav-item');
                    if (remainingItems.length > 0) {
                        const firstLink = remainingItems[0].querySelector('.nav-link');
                        if (firstLink) firstLink.classList.add('active');
                    }
                }
                const confirmModal = document.getElementById('confirmModal');
                if (confirmModal && window.bootstrap) {
                    const modal = bootstrap.Modal.getInstance(confirmModal);
                    if (modal) modal.hide();
                }
            });
        }
    }

    function loadChatHistory(date) {
        if (!chatHistory) return;
        chatHistory.innerHTML = '';
        fetch(`/api/history/${date}`)
            .then(res => res.json())
            .then(data => {
                data.messages.forEach(msg => {
                    addMessageToChat(msg.sender, msg.text);
                });
            })
            .catch(error => console.error('Error loading history:', error));
    }

    function restoreSidebarState() {
        if (!sidebar) return;
        if (typeof(Storage) !== "undefined") {
            const isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
            if (isCollapsed) sidebar.classList.add('collapsed');
        }
    }

    function initializeTooltips() {
        if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(el => new bootstrap.Tooltip(el));
        }
    }

    function setupResponsiveHandling() {
        window.addEventListener('resize', function () {
            if (window.innerWidth <= 768 && sidebar) sidebar.classList.remove('collapsed');
        });
    }

    function init() {
        initializeFAQ();
        setupTextareaResize();
        setupEventListeners();
        setupDeleteConfirmation();
        restoreSidebarState();
        initializeTooltips();
        setupResponsiveHandling();
    }

    init();
});

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;
    sidebar.classList.toggle('collapsed');
    if (typeof(Storage) !== "undefined") {
        const isCollapsed = sidebar.classList.contains('collapsed');
        localStorage.setItem('sidebarCollapsed', isCollapsed);
    }
}

function addToSidebar(message) {
    const sidebarHistory = document.getElementById("chat-history-sidebar");
    if (!sidebarHistory) return;

    const li = document.createElement("li");
    li.className = "list-group-item small nav-item";
    li.innerHTML = `
        <a href="#" class="nav-link">
            <span class="nav-text">${message.substring(0, 30)}...</span>
            <span class="delete-btn float-end text-danger" title="Delete">&times;</span>
        </a>`;
    sidebarHistory.appendChild(li);
}

const sidebarToggle = document.getElementById("sidebarToggle");
if (sidebarToggle) {
    sidebarToggle.addEventListener("click", function () {
        toggleSidebar();
        sidebarToggle.classList.toggle("sidebar-open");
    });
}
