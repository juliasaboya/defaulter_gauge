const filters = document.querySelector('#filtros');
const search = document.querySelector('#busca');
const category = document.querySelector('#categoria');
const status = document.querySelector('#status');
const normalize = (value) => value.toLowerCase().normalize('NFD').replace(/\p{Mark}/gu, '');

const entries = Array.from(document.querySelectorAll('#termos .entry'), (element) => ({
  element,
  category: element.dataset.category,
  text: normalize(element.textContent),
}));

function filterTerms() {
  const words = normalize(search.value).trim().split(/\s+/u).filter(Boolean);
  let count = 0;
  for (const entry of entries) {
    const matches = (!category.value || entry.category === category.value)
      && words.every((word) => entry.text.includes(word));
    entry.element.hidden = !matches;
    count += Number(matches);
  }
  status.textContent = count === 0
    ? 'Nenhum termo encontrado. Experimente outra palavra ou categoria.'
    : `${count} ${count === 1 ? 'termo' : 'termos'}`;
}

search.addEventListener('input', filterTerms);
category.addEventListener('change', filterTerms);
filters.hidden = false;
filterTerms();
