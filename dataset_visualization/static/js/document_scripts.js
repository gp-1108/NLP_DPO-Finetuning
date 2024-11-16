document.addEventListener('DOMContentLoaded', () => {
    const toggleButtons = document.querySelectorAll('.toggle-btn');

    toggleButtons.forEach(button => {
        button.addEventListener('click', () => {
            const listId = button.getAttribute('data-list');
            const list = document.getElementById(listId);

            if (list.classList.contains('expanded')) {
                list.classList.remove('expanded');
                button.textContent = 'Expand';
            } else {
                list.classList.add('expanded');
                button.textContent = 'Collapse';
            }
        });
    });
});
