document.getElementById("send-btn").addEventListener("click", function() {
    var messageInput = document.getElementById("message-input");
    var message = messageInput.value;
    
    if (message.trim() !== "") {
      var chatMessages = document.getElementById("chat-messages");
  
      // User message
      var userMessageDiv = document.createElement("div");
      userMessageDiv.classList.add("message", "user-message", "bg-success", "text-white", "p-3", "mb-2", "rounded");
      userMessageDiv.innerHTML = `<p>${message}</p><small class="text-muted d-block text-right">10:02 AM</small>`;
      chatMessages.appendChild(userMessageDiv);
  
      // Scroll to bottom
      chatMessages.scrollTop = chatMessages.scrollHeight;
  
      // Clear input
      messageInput.value = "";
  
      // Simulate bot response after a delay
      setTimeout(function() {
        var botMessageDiv = document.createElement("div");
        botMessageDiv.classList.add("message", "bot-message", "bg-light", "text-dark", "p-3", "mb-2", "rounded");
        botMessageDiv.innerHTML = `<p>Thanks for your message!</p><small class="text-muted d-block text-left">10:03 AM</small>`;
        chatMessages.appendChild(botMessageDiv);
  
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
      }, 1000);
    }
  });
  