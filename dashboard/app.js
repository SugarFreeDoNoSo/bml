const search = document.querySelector('#search');
const state = document.querySelector('#state');
function filterPlans() {
  const query = search.value.trim().toLocaleLowerCase('es');
  let visible = 0;
  document.querySelectorAll('.card').forEach(card => {
    const nameMatches = card.dataset.name.includes(query);
    let matches = 0;
    card.querySelectorAll('li').forEach(task => {
      task.hidden = !(state.value === 'all' || task.dataset.state === state.value)
        || !(nameMatches || task.textContent.toLocaleLowerCase('es').includes(query));
      if (!task.hidden) matches++;
    });
    card.hidden = matches === 0;
    card.querySelector('.empty').hidden = matches !== 0;
    if (!card.hidden) visible++;
    if (query || state.value !== 'all') card.querySelector('details').open = true;
  });
  document.querySelector('#no-results').hidden = visible !== 0;
}
search.addEventListener('input', filterPlans);
state.addEventListener('change', filterPlans);
