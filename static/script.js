document.addEventListener("DOMContentLoaded", function () {
    // DOM Elements
    const elements = {
        textarea: document.getElementById("chat-input"),
        sendBtn: document.getElementById("send-btn"),
        chatHistory: document.getElementById("chat-history"),
        faqSuggestions: document.getElementById("faq-suggestions"),
        faqList: document.getElementById("faq-list"),
        sidebar: document.getElementById("sidebar"),
        historyDateList: document.getElementById('historyDateList'),
        sidebarToggle: document.getElementById('sidebarToggle'),
        confirmModal: document.getElementById('confirmModal'),
        loadingOverlay: document.getElementById('chat-loading'),
        confirmDelete: document.getElementById('confirmDelete')
    };

    // State
    const state = {
        historyToDelete: null,
        datesData: [],
        mostAskedQuestions: [
            "What is the company vision?",
            "Tell me about our core values",
            "What types of contracts are available?",
            "How do I apply for leave?",
            "What are the unacceptable actions?",
            "Can you explain the disciplinary procedure?"
        ]
    };

    // Initialize FAQ suggestions
    function initializeFAQ() {
        if (!elements.faqList) return;

        // Clear existing items
        elements.faqList.innerHTML = '';

        // Create new FAQ items
        state.mostAskedQuestions.forEach(question => {
            const listItem = document.createElement('li');
            listItem.className = 'faq-item mb-2';

            const button = document.createElement('button');
            button.className = 'btn btn-outline-primary btn-sm faq-btn text-start';
            button.textContent = question;

            listItem.appendChild(button);
            elements.faqList.appendChild(listItem);
        });

        // Make sure FAQ section is visible by default
        toggleFAQVisibility(true);
    }

    // Handle user input
    function handleUserInput() {
        const message = elements.textarea.value.trim();
        if (!message) return;

        addMessageToChat('user', message);
        elements.textarea.value = "";
        sendToAPI(message);
        toggleFAQVisibility(false);
    }

    // API communication
    function sendToAPI(message) {

        const typingBubble = createTypingIndicator();
        elements.chatHistory.appendChild(typingBubble);
        scrollToBottom();

        fetch("/api/respond", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: message })
        })
            .then(response => response.json())
            .then(handleAPIResponse)
            .catch(handleAPIError)
            .finally(hideLoading);
    }

    function handleAPIResponse(data) {
        // Remove typing indicator
        const typingIndicators = document.querySelectorAll('.typing-indicator');
        typingIndicators.forEach(indicator => indicator.remove());

        if (data.response) {
            addMessageToChat('bot', data.response);
        }
    }

    function handleAPIError(error) {
        console.error("API Error:", error);
        addMessageToChat('bot', "Sorry, I encountered an error. Please try again later.");
    }

    function createTypingIndicator() {
        const div = document.createElement("div");
        div.className = "chat-bubble bot-message typing-indicator";
        div.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2"></span>
            Bot is responding ...
        `;
        return div;
    }

    // Chat message handling
    function addMessageToChat(sender, message, timestamp = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-bubble ${sender}-message`;

        // Use provided timestamp or current time if not provided
        const displayTime = timestamp ?
            new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) :
            new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        messageDiv.innerHTML = `
        <div class="message-content">${message}</div>
        <div class="message-meta">${displayTime}</div>
    `;

        elements.chatHistory.appendChild(messageDiv);
        scrollToBottom();
    }

    function scrollToBottom() {
        elements.chatHistory.scrollTop = elements.chatHistory.scrollHeight;
    }

    // FAQ visibility
    function toggleFAQVisibility(show) {
        if (!elements.faqSuggestions) return;

        elements.faqSuggestions.style.display = show ? "block" : "none";

        // Also toggle a class for more control if needed
        if (show) {
            elements.faqSuggestions.classList.add('visible');
            elements.faqSuggestions.classList.remove('hidden');
        } else {
            elements.faqSuggestions.classList.add('hidden');
            elements.faqSuggestions.classList.remove('visible');
        }
    }

    // Loading states
    function showLoading() {
        if (elements.loadingOverlay) elements.loadingOverlay.style.display = 'flex';
    }

    function hideLoading() {
        if (elements.loadingOverlay) elements.loadingOverlay.style.display = 'none';
    }

    // Sidebar functionality
    function setupSidebar() {
        if (!elements.sidebarToggle || !elements.sidebar) return;

        const overlay = document.querySelector('.sidebar-overlay');

        // Toggle sidebar
        function toggleSidebar() {
            if (window.innerWidth <= 768) {
                // Mobile behavior
                elements.sidebar.classList.toggle('show');
                overlay.style.display = elements.sidebar.classList.contains('show') ? 'block' : 'none';
                document.body.style.overflow = elements.sidebar.classList.contains('show') ? 'hidden' : '';
            } else {
                // Desktop behavior
                elements.sidebar.classList.toggle('collapsed');
                if (typeof Storage !== "undefined") {
                    localStorage.setItem('sidebarCollapsed', elements.sidebar.classList.contains('collapsed'));
                }
            }
        }

        // Event listeners
        elements.sidebarToggle.addEventListener('click', function (e) {
            e.stopPropagation();
            toggleSidebar();
        });

        overlay.addEventListener('click', toggleSidebar);

        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', function (event) {
            if (window.innerWidth <= 768 &&
                !elements.sidebar.contains(event.target) &&
                !elements.sidebarToggle.contains(event.target) &&
                elements.sidebar.classList.contains('show')) {
                toggleSidebar();
            }
        });

        // Handle window resize
        function handleResize() {
            if (window.innerWidth > 768) {
                // Reset mobile states when switching to desktop
                elements.sidebar.classList.remove('show');
                overlay.style.display = 'none';
                document.body.style.overflow = '';
            }
        }

        window.addEventListener('resize', handleResize);
    }

    // Initialize the application
    function init() {
        initializeFAQ();
        setupSidebar();
        setupEventListeners();
        loadHistoryDates();
        restoreSidebarState();

        // Initialize Bootstrap modal
        if (elements.confirmModal) {
            state.modal = new bootstrap.Modal(elements.confirmModal);
        }
    }

    // Event listeners setup
    function setupEventListeners() {
        // Send message
        if (elements.sendBtn) {
            elements.sendBtn.addEventListener("click", handleUserInput);
        }

        // Textarea input
        if (elements.textarea) {
            elements.textarea.addEventListener("keydown", function (e) {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleUserInput();
                }
            });
        }

        // FAQ suggestions
        if (elements.faqList) {
            elements.faqList.addEventListener("click", function (e) {
                if (e.target.classList.contains("faq-btn")) {
                    const question = e.target.textContent;
                    // Add to chat and send immediately
                    addMessageToChat('user', question);
                    sendToAPI(question);
                    toggleFAQVisibility(false);
                }
            });
        }

        // Delete confirmation
        if (elements.confirmDelete) {
            elements.confirmDelete.addEventListener("click", function () {
                if (state.historyToDelete) {
                    const date = state.historyToDelete.getAttribute('data-date');
                    deleteChatHistory(date);
                    state.modal.hide();
                }
            });
        }
    }

    // Chat history functions
    function loadHistoryDates() {
        fetch('/api/history/unique-dates')
            .then(res => {
                if (!res.ok) throw new Error('Network response was not ok');
                return res.json();
            })
            .then(data => {
                if (data.status === 'success' && Array.isArray(data.dates)) {
                    state.datesData = data.dates;
                    renderDateList();
                }
            })
            .catch(error => {
                console.error('Error loading history dates:', error);
            });
    }

    function renderDateList() {
        if (!elements.historyDateList) return;

        elements.historyDateList.innerHTML = state.datesData.map(date => `
            <li class="nav-item">
                <a href="#" class="nav-link" data-date="${date}">
                    <span class="nav-text">${date}</span>
                    <button class="delete-btn" title="Delete ${date}">
                        <i class="fas fa-trash"></i>
                    </button>
                </a>
            </li>
        `).join('');

        // Event delegation for history items
        elements.historyDateList.addEventListener('click', function (e) {
            const deleteBtn = e.target.closest('.delete-btn');
            const navLink = e.target.closest('.nav-link');

            if (deleteBtn) {
                e.preventDefault();
                e.stopPropagation();
                state.historyToDelete = navLink;
                state.modal.show();
            } else if (navLink) {
                e.preventDefault();
                loadChatHistory(navLink.getAttribute('data-date'));
            }
        });
    }

    function loadChatHistory(date) {
        showLoading();
        elements.chatHistory.innerHTML = '';

        fetch(`/api/history/${date}`)
            .then(res => {
                if (!res.ok) throw new Error('Network response was not ok');
                return res.json();
            })
            .then(data => {
                if (data.status === 'success' && Array.isArray(data.data)) {
                    data.data.forEach(item => {
                        if (item.user_input) addMessageToChat('user', item.user_input, item.timestamp);
                        if (item.bot_response) addMessageToChat('bot', item.bot_response, item.timestamp);
                    });
                }
            })
            .catch(error => {
                console.error('Error loading chat history:', error);
                addMessageToChat('bot', "Failed to load chat history. Please try again.");
            })
            .finally(hideLoading);
    }

    function deleteChatHistory(date) {
        fetch(`/api/history/${date}`, { method: 'DELETE' })
            .then(res => {
                if (!res.ok) throw new Error('Network response was not ok');
                return res.json();
            })
            .then(data => {
                if (data.status === 'success') {
                    loadHistoryDates(); // Refresh the list
                }
            })
            .catch(error => {
                console.error('Error deleting chat history:', error);
            });
    }

    // Restore sidebar state
    function restoreSidebarState() {
        if (window.innerWidth <= 768) return; // Only for desktop

        if (typeof Storage !== "undefined") {
            const isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
            if (isCollapsed) elements.sidebar.classList.add('collapsed');
        }
    }

    // Initialize the app
    init();
});