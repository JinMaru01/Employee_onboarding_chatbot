document.addEventListener("DOMContentLoaded", function () {
    // DOM Elements
    const textarea = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const chatHistory = document.getElementById("chat-history");
    const faqSuggestions = document.getElementById("faq-suggestions");
    const faqList = document.getElementById("faq-list");
    const sidebar = document.getElementById("sidebar");
    const historyDateList = document.getElementById('historyDateList');
    const chatHistoryDiv = document.getElementById('chat-history');

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
    let datesData = [];

    function initializeFAQ() {
        if (faqSuggestions && faqList) {
            faqSuggestions.style.display = "block";
            faqList.innerHTML = "";
            mostAskedQuestions.forEach(question => {
                const li = document.createElement("li");
                li.className = "faq-item mb-2";
                li.innerHTML = `<button class="btn btn-outline-primary btn-sm faq-btn">${question}</button>`;
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
        if (!chatHistoryDiv) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-bubble ${sender === 'user' ? 'user-message' : 'bot-message'}`;
        messageDiv.textContent = message;
        chatHistoryDiv.appendChild(messageDiv);
        chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
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
    }

    function showConfirmModal() {
        const modal = document.getElementById('confirmModal');
        if (modal && typeof bootstrap !== 'undefined') {
            const bsModal = new bootstrap.Modal(modal);
            bsModal.show();
        }
    }

    function hideConfirmModal() {
        const modal = document.getElementById('confirmModal');
        if (modal && typeof bootstrap !== 'undefined') {
            const bsModal = bootstrap.Modal.getInstance(modal);
            if (bsModal) {
                bsModal.hide();
            }
        }
    }

    function setupDeleteConfirmation() {
        const confirmDelete = document.getElementById('confirmDelete');
        if (confirmDelete) {
            confirmDelete.addEventListener('click', function () {
                if (historyToDelete) {
                    const dateToDelete = historyToDelete.getAttribute('data-date');

                    // Remove from UI
                    historyToDelete.remove();

                    // Remove from datesData array
                    datesData = datesData.filter(date => date !== dateToDelete);

                    // Clear chat history if this was the active date
                    const activeDateElement = document.querySelector('.nav-link.active');
                    if (!activeDateElement || activeDateElement === historyToDelete) {
                        chatHistoryDiv.innerHTML = '';
                        showFAQSuggestions();
                        addWelcomeMessage();

                        // Set first remaining date as active if any exist
                        if (datesData.length > 0) {
                            setActiveDate(datesData[0]);
                            loadChatHistory(datesData[0]);
                        }
                    }

                    fetch(`/api/history/${dateToDelete}`, { method: 'DELETE' })
                        .then(response => response.json())
                        .then(data => {
                            console.log('Deleted:', data);
                            if (data.status !== 'success') {
                                alert('Failed to delete from server: ' + data.message);
                            }
                        })
                        .catch(error => {
                            console.error('Delete error:', error);
                            alert('Server error while deleting chat history.');
                        });
                }
                historyToDelete = null;
                hideConfirmModal();
            });
        }

        // Handle modal cancel/close
        const confirmModal = document.getElementById('confirmModal');
        if (confirmModal) {
            confirmModal.addEventListener('hidden.bs.modal', function () {
                historyToDelete = null;
            });
        }
    }

    function loadChatHistory(date) {
        const loader = document.getElementById('chat-loading');
        const chatHistoryDiv = document.getElementById('chat-history');
        if (!chatHistoryDiv || !loader) return;

        chatHistoryDiv.innerHTML = '';
        loader.style.display = 'flex';

        fetch(`/api/history/${date}`)
            .then(res => res.json())
            .then(data => {
                loader.style.display = 'none';

                if (data.status === 'success' && Array.isArray(data.data)) {
                    data.data.forEach(item => {
                        if (item.user_input) addMessageToChat('user', item.user_input);
                        if (item.bot_response) addMessageToChat('bot', item.bot_response);
                    });
                } else {
                    chatHistoryDiv.innerHTML = '<p><em>Failed to load chat history.</em></p>';
                }
            })
            .catch(error => {
                loader.style.display = 'none';
                console.error('Error:', error);
                chatHistoryDiv.innerHTML = '<p><em>Error loading chat history.</em></p>';
            });
    }

    function restoreSidebarState() {
        if (!sidebar) return;
        if (typeof (Storage) !== "undefined") {
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

    // Load history dates and render sidebar
    function loadHistoryDates() {
        fetch('/api/history/unique-dates')
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success' && Array.isArray(data.dates)) {
                    datesData = data.dates;
                    renderDateList();
                    showFAQSuggestions();
                    addWelcomeMessage();
                } else {
                    console.error('Invalid date list response:', data);
                    showFAQSuggestions();
                    addWelcomeMessage();
                }
            })
            .catch(error => {
                console.error('Failed to load dates:', error);
                showFAQSuggestions();
                addWelcomeMessage();
            });
    }

    function renderDateList() {
        if (!historyDateList) return;

        historyDateList.innerHTML = '';
        datesData.forEach(date => {
            const li = document.createElement('li');
            li.className = 'nav-item';

            li.innerHTML = `
                <a href="#" class="nav-link" data-date="${date}">
                    <span class="nav-text">${date}</span>
                    <button class="delete-btn btn btn-link text-danger p-0" title="Delete ${date}" type="button">
                        <i class="fas fa-trash"></i>
                    </button>
                </a>
            `;

            // Click event to load chat history for this date
            const navLink = li.querySelector('a.nav-link');
            navLink.addEventListener('click', (e) => {
                e.preventDefault();
                if (e.target.closest('.delete-btn')) return; // Don't load history if delete button clicked

                const clickedDate = e.currentTarget.getAttribute('data-date');
                setActiveDate(clickedDate);
                loadChatHistory(clickedDate);
            });

            // Delete button event
            const deleteBtn = li.querySelector('.delete-btn');
            deleteBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                historyToDelete = navLink;
                showConfirmModal();
            });

            historyDateList.appendChild(li);
        });
    }

    function setActiveDate(date) {
        if (!historyDateList) return;
        [...historyDateList.querySelectorAll('a.nav-link')].forEach(link => {
            link.classList.toggle('active', link.getAttribute('data-date') === date);
        });
    }

    function showFAQSuggestions() {
        const faqSuggestions = document.getElementById("faq-suggestions");
        const faqList = document.getElementById("faq-list");

        if (faqSuggestions && faqList) {
            faqSuggestions.style.display = "block";
            faqList.innerHTML = "";
            mostAskedQuestions.forEach(question => {
                const li = document.createElement("li");
                li.className = "faq-item mb-2";
                li.innerHTML = `<button class="btn btn-outline-primary btn-sm faq-btn">${question}</button>`;
                faqList.appendChild(li);
            });
        }
    }

    function addWelcomeMessage() {
        if (!chatHistoryDiv) return;
        if (chatHistoryDiv.children.length > 0) return;

        const welcomeText = "👋 Welcome! How can I help you today? Here are some popular questions to get started.";
        const messageDiv = document.createElement('div');
        messageDiv.className = "chat-bubble bot-message";
        messageDiv.textContent = welcomeText;
        chatHistoryDiv.appendChild(messageDiv);
        chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
    }

    function init() {
        initializeFAQ();
        setupTextareaResize();
        setupEventListeners();
        setupDeleteConfirmation();
        restoreSidebarState();
        initializeTooltips();
        setupResponsiveHandling();
        loadHistoryDates();
    }

    init();
});

// Sidebar toggle functionality
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;
    sidebar.classList.toggle('collapsed');
    if (typeof (Storage) !== "undefined") {
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

// Initialize sidebar toggle
const sidebarToggle = document.getElementById("sidebarToggle");
if (sidebarToggle) {
    sidebarToggle.addEventListener("click", function () {
        toggleSidebar();
        sidebarToggle.classList.toggle("sidebar-open");
    });
}

function showLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.style.display = 'flex';
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.style.display = 'none';
}