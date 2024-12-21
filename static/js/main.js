// main.js
let currentAudio = null;
let isListening = false;
let recognition = null;
let currentImage = null;
let currentLang = 'en-US';

function showLoading() {
    document.getElementById('loadingIndicator').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loadingIndicator').classList.add('hidden');
}

function showError(message) {
    addMessage('assistant', `Error: ${message}`, new Date().toLocaleTimeString());
}

function stopAllAudio() {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
        currentAudio = null;
    }
    document.querySelectorAll('audio').forEach(audio => {
        audio.pause();
        audio.currentTime = 0;
    });
}

function formatContent(content) {
    if (!content) return '';
    
    content = content.replace(/(\d+\.\s*\*\*[\w\s]+:\*\*)/g, '<div class="mt-3 mb-2">$1</div>');
    content = content.replace(/\*\*(.*?)\*\*/g, '<strong class="text-green-700">$1</strong>');
    
    const paragraphs = content.split('\n');
    return paragraphs.map(para => {
        if (para.trim() === '') return '';
        if (para.match(/^\d+\./)) {
            return `<div class="ml-4 my-2">${para}</div>`;
        }
        return `<p class="mb-2">${para}</p>`;
    }).join('');
}

function addMessage(role, content, timestamp, audioUrl = null, voice_mode = false) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    const isUser = role === 'user';
    
    messageDiv.className = `flex ${isUser ? 'justify-end' : 'justify-start'}`;
    
    let audioHtml = '';
    if (audioUrl && voice_mode && !isUser) {
        audioHtml = `
            <div class="mt-2">
                <audio controls>
                    <source src="${audioUrl}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
            </div>
        `;
    }
    
    const formattedContent = isUser ? content : formatContent(content);
    
    messageDiv.innerHTML = `
        <div class="max-w-[80%]">
            <div class="message-bubble rounded-lg px-4 py-2 ${
                isUser ? 'bg-gradient-to-r from-green-500 to-teal-500 text-white' : 'bg-gradient-to-br from-green-50 to-teal-50 text-gray-800'
            }">
                <div class="${isUser ? '' : 'prose prose-green max-w-none'}">
                    ${formattedContent}
                </div>
                ${audioHtml}
            </div>
            <div class="text-xs text-gray-500 mt-1 ${isUser ? 'text-right' : 'text-left'}">
                ${timestamp}
            </div>
        </div>
    `;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    if (audioUrl && voice_mode && !isUser) {
        const audioElement = messageDiv.querySelector('audio');
        if (audioElement) {
            if (currentAudio) {
                currentAudio.pause();
            }
            currentAudio = audioElement;
            audioElement.play().catch(e => console.error("Audio playback error:", e));
        }
    }
}

async function sendMessage(fromVoice = false) {
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    
    if (!message && !currentImage) return;
    
    addMessage('user', message || 'Analyzing image...', new Date().toLocaleTimeString());
    
    input.value = '';
    input.disabled = true;
    showLoading();
    
    try {
        const requestData = {
            query: message,
            voice_mode: fromVoice,
            language: currentLang.split('-')[0]
        };

        if (currentImage) {
            requestData.image = currentImage;
        }

        const response = await fetch('/process_query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
        } else {
            if (currentImage) {
                clearImageUpload();
            }
            
            addMessage(
                'assistant',
                data.response,
                new Date().toLocaleTimeString(),
                data.audio_url,
                fromVoice
            );
        }
        
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to process request');
    }
    
    hideLoading();
    input.disabled = false;
    input.focus();
}

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('previewImg').src = e.target.result;
            document.getElementById('imagePreview').classList.remove('hidden');
            currentImage = e.target.result;
        }
        reader.readAsDataURL(file);
    }
}

function clearImageUpload() {
    document.getElementById('imageUpload').value = '';
    document.getElementById('imagePreview').classList.add('hidden');
    document.getElementById('previewImg').src = '';
    currentImage = null;
}


function toggleLanguage() {
    currentLang = currentLang === 'en-US' ? 'ur-PK' : 'en-US';
    
    const langButton = document.getElementById('langButton');
    const langText = langButton.querySelector('.lang-text');
    
    // Update button text
    langText.textContent = currentLang === 'en-US' ? 'EN' : 'اردو';
    
    // Update input placeholder
    const input = document.getElementById('userInput');
    input.placeholder = currentLang === 'en-US' ? 
        'Ask about crops, weather, market prices...' : 
        'فصلوں، موسم، مارکیٹ کے بارے میں پوچھیں...';
    
    // Update speech recognition language
    if (recognition) {
        recognition.stop();  // Stop any ongoing recognition
        recognition.lang = currentLang === 'ur-PK' ? 'ar-SA' : 'en-US';  // Use Arabic as fallback for Urdu
    }
    
    // Show language change feedback
    showLanguageToast(currentLang === 'ur-PK' ? 'اردو موڈ فعال ہے' : 'English Mode Active');
}
// Add this function for showing toast notifications
function showLanguageToast(message) {
    // Remove existing toast if any
    const existingToast = document.querySelector('.language-toast');
    if (existingToast) {
        existingToast.remove();
    }
    
    // Create new toast
    const toast = document.createElement('div');
    toast.className = 'language-toast fixed bottom-4 right-4 bg-green-500 text-white px-4 py-2 rounded-lg shadow-lg transition-opacity duration-300';
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    // Remove after 2 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 2000);
}

function initializeSpeechRecognition() {
    if ('webkitSpeechRecognition' in window) {  // Use webkit prefix specifically
        recognition = new webkitSpeechRecognition();
        
        // Configure for better Urdu support
        recognition.continuous = false;
        recognition.interimResults = true;  // Enable interim results
        recognition.maxAlternatives = 3;    // Get multiple alternatives
        
        // Set language based on toggle but with broader regional support
        recognition.lang = currentLang === 'ur-PK' ? 'ar-SA' : 'en-US';  // Try Arabic as fallback for Urdu
        
        recognition.onstart = function() {
            isListening = true;
            updateVoiceButton(true);
            const input = document.getElementById('userInput');
            input.placeholder = currentLang === 'ur-PK' ? 'بولیں...' : 'Speaking...';
        };
        
        recognition.onend = function() {
            isListening = false;
            updateVoiceButton(false);
            const input = document.getElementById('userInput');
            input.placeholder = currentLang === 'ur-PK' ? 
                'فصلوں، موسم، مارکیٹ کے بارے میں پوچھیں...' : 
                'Ask about crops, weather, market prices...';
        };
        
        recognition.onresult = function(event) {
            let finalTranscript = '';
            let interimTranscript = '';
            
            // Process all results to get the best match
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }
            
            // Show interim results while speaking
            if (interimTranscript !== '') {
                document.getElementById('userInput').value = interimTranscript;
            }
            
            // Process final result
            if (finalTranscript !== '') {
                document.getElementById('userInput').value = finalTranscript;
                // Small delay to ensure the final transcript is complete
                setTimeout(() => {
                    sendMessage(true);
                }, 500);
            }
        };
        
        recognition.onerror = function(event) {
            console.error('Speech recognition error:', event.error);
            isListening = false;
            updateVoiceButton(false);
            
            // More specific error messages
            let errorMessage;
            switch(event.error) {
                case 'no-speech':
                    errorMessage = currentLang === 'ur-PK' ? 
                        'کوئی آواز نہیں سنی گئی' : 
                        'No speech detected';
                    break;
                case 'audio-capture':
                    errorMessage = currentLang === 'ur-PK' ? 
                        'مائیکروفون کی رسائی دستیاب نہیں' : 
                        'No microphone access';
                    break;
                case 'not-allowed':
                    errorMessage = currentLang === 'ur-PK' ? 
                        'مائیکروفون کی اجازت درکار ہے' : 
                        'Microphone permission needed';
                    break;
                default:
                    errorMessage = currentLang === 'ur-PK' ? 
                        'آواز کی شناخت میں خرابی' : 
                        'Speech recognition error';
            }
            showError(errorMessage);
        };

        // Add language detection feedback
        recognition.onlanguagechange = function(event) {
            console.log('Language changed to:', recognition.lang);
        };
    }
}

function updateVoiceButton(isRecording) {
    const voiceButton = document.getElementById('voiceButton');
    if (isRecording) {
        voiceButton.innerHTML = '<i class="fas fa-stop"></i>';
        voiceButton.classList.add('animate-pulse', 'bg-red-500');
        voiceButton.classList.remove('bg-green-500');
    } else {
        voiceButton.innerHTML = '<i class="fas fa-microphone"></i>';
        voiceButton.classList.remove('animate-pulse', 'bg-red-500');
        voiceButton.classList.add('bg-green-500');
    }
}

function toggleVoiceInput() {
    if (!recognition) {
        initializeSpeechRecognition();
    }

    if (!isListening) {
        stopAllAudio();
        try {
            recognition.start();
        } catch (error) {
            console.error('Error starting speech recognition:', error);
            showError('Could not start speech recognition');
        }
    } else {
        recognition.stop();
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    
    // Initialize speech recognition
    initializeSpeechRecognition();
    
    // Keyboard events
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage(false);
        }
    });
    
    // Button clicks
    sendButton.addEventListener('click', () => sendMessage(false));
    
    // Input focus handling
    input.addEventListener('focus', () => {
        input.classList.add('ring-2', 'ring-green-500');
        stopAllAudio();
    });
    
    input.addEventListener('blur', () => {
        input.classList.remove('ring-2', 'ring-green-500');
    });
    
    // Category button handling
    document.querySelectorAll('.category-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const category = this.querySelector('span').textContent;
            input.value = `Tell me about ${category.toLowerCase()}`;
            input.focus();
        });
    });
});

// Global keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        if (isListening) {
            recognition.stop();
        }
        if (currentAudio) {
            stopAllAudio();
        }
    }
});