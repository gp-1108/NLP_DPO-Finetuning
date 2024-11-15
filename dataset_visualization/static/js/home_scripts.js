document.addEventListener('DOMContentLoaded', () => {
    const documentsList = document.getElementById('documents-list');
    const dialoguesList = document.getElementById('dialogues-list');
    const dpoDialoguesList = document.getElementById('dpo-dialogues-list');

    // Add click event listener to filter buttons
    document.querySelectorAll('.filter-btn').forEach(button => {
        button.addEventListener('click', event => {
            event.stopPropagation(); // Prevent clicking the parent item
            const selectedId = button.getAttribute('data-id');
            const type = button.getAttribute('data-type');

            if (type === 'document') {
                filterList(dialoguesList, selectedId);
                filterList(dpoDialoguesList, selectedId);
            } else if (type === 'dialogue') {
                filterList(dpoDialoguesList, selectedId);
            }
        });
    });

    // Reset filters when clicking outside lists
    document.body.addEventListener('click', event => {
        if (!event.target.classList.contains('filter-btn')) {
            resetList(dialoguesList);
            resetList(dpoDialoguesList);
        }
    });

    /**
     * Filters the list by hiding items that don't contain the substring.
     * @param {HTMLElement} list - The list to filter.
     * @param {string} substring - The substring to filter by.
     */
    function filterList(list, substring) {
        const items = list.querySelectorAll('.list-item');
        items.forEach(item => {
            const itemId = item.querySelector('.item-link').textContent;
            if (itemId.includes(substring)) {
                item.style.display = 'flex';
            } else {
                item.style.display = 'none';
            }
        });
    }

    /**
     * Resets the list by showing all items.
     * @param {HTMLElement} list - The list to reset.
     */
    function resetList(list) {
        const items = list.querySelectorAll('.list-item');
        items.forEach(item => {
            item.style.display = 'flex';
        });
    }
});
