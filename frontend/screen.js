   // Redirect flag to prevent multiple redirects
        let isRedirecting = false;
        
        // Function to redirect to the main page
        function redirectToMain() {
            if (isRedirecting) return;
            isRedirecting = true;
            
            document.body.classList.add('fade-out');
            
            setTimeout(() => {
                try {
                    window.location.href = 'web.html';
                } catch (error) {
                    console.error('Redirect error:', error);
                    // Fallback in case of error
                    document.body.innerHTML = '<div style="display: flex; justify-content: center; align-items: center; height: 100vh; font-family: sans-serif;">' +
                        '<div style="text-align: center; padding: 20px;">' +
                        '<h2>Redirect Error</h2>' +
                        '<p>Terjadi kesalahan saat memuat aplikasi. Silakan refresh halaman atau klik <a href="web.html">di sini</a>.</p>' +
                        '</div></div>';
                }
            }, 800); // Wait for fade-out animation to complete
        }

        // Auto transition after loading completes
        setTimeout(redirectToMain, 5500);

        // Optional: Click to skip
        document.addEventListener('click', redirectToMain);

        // Optional: Tap to skip on touch devices
        document.addEventListener('touchstart', redirectToMain);

        // Prevent context menu for professional look
        document.addEventListener('contextmenu', (e) => {
            e.preventDefault();
        });