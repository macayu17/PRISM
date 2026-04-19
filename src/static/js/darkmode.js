document.addEventListener('DOMContentLoaded', function() {
    const themeToggle = document.getElementById('theme-toggle');
    const htmlElement = document.documentElement;
    
    // Check for saved theme preference or use device preference
    const savedTheme = localStorage.getItem('theme') || 
                      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    
    // Apply saved theme
    htmlElement.setAttribute('data-theme', savedTheme);
    updateToggleIcon(savedTheme);
    
    // Toggle theme when button is clicked
    themeToggle.addEventListener('click', function() {
        const currentTheme = htmlElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        htmlElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateToggleIcon(newTheme);
    });
    
    // Update toggle icon based on current theme
    function updateToggleIcon(theme) {
        if (theme === 'dark') {
            themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
            themeToggle.title = 'Switch to light mode';
        } else {
            themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
            themeToggle.title = 'Switch to dark mode';
        }
    }
    
    // Initialize tooltips
    initializeTooltips();
});

function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(element => {
        const tooltipText = element.getAttribute('data-tooltip');
        const tooltipSpan = document.createElement('span');
        tooltipSpan.className = 'tooltiptext';
        tooltipSpan.textContent = tooltipText;
        
        element.classList.add('tooltip');
        element.appendChild(tooltipSpan);
    });
}