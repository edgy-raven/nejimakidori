import {tileElement} from './tiles.js';
import {playbackState} from './replay.js';
import {englishLabel, winText, winPoints} from './labels.js';

function indicatorRows(dora, ura = []) {
  const indicators = document.createElement('div');
  indicators.className = 'kyoku-indicators';
  for (const [name, tiles] of [['Dora', dora], ['Ura', ura]]) {
    const row = document.createElement('div');
    row.className = 'kyoku-indicator-row';
    row.setAttribute('aria-label', `${name} indicators`);
    for (let index = 0; index < 5; index++) {
      const tile = tileElement(tiles[index] || '?');
      if (!tiles[index]) {
        tile.textContent = '';
        tile.classList.add('tile-back');
      }
      row.append(tile);
    }
    indicators.append(row);
  }
  return indicators;
}

export function kyokuNotice(board, controls = document, {closeOnDiscard = false} = {}) {
  let currentKey;
  let popup;
  let previousRound;
  let previousLabel;
  const dismiss = () => {
    popup?.remove();
    popup = null;
  };
  controls.addEventListener('click', event => {
    if (event.target.closest('button, input, select, [data-mistake-decision], .hand')) dismiss();
  }, true);
  controls.addEventListener('input', dismiss, true);
  controls.addEventListener('change', dismiss, true);
  controls.addEventListener('keydown', event => {
    if (['ArrowLeft', 'ArrowRight', ' '].includes(event.key)
        && !event.target.matches('input, select')) dismiss();
  }, true);
  return (key, label, round, selfSeat, {show = true} = {}) => {
    if (key !== currentKey) {
      dismiss();
      if (show && currentKey !== undefined) {
        popup = document.createElement('section');
        popup.className = 'kyoku-notice';
        popup.setAttribute('aria-label', 'Hand result');
        const title = document.createElement('strong');
        title.setAttribute('role', 'status');
        title.textContent = previousLabel;
        const close = document.createElement('button');
        close.type = 'button';
        close.setAttribute('aria-label', 'Close hand result');
        close.textContent = '×';
        close.addEventListener('click', dismiss);
        const heading = document.createElement('header');
        heading.append(title, close);
        popup.append(heading);
        const result = previousRound.events.findLast(event =>
          ['agari', 'ryuukyoku'].includes(event.type));
        if (result) {
          const state = playbackState(previousRound);
          popup.prepend(indicatorRows(state.winners[0]?.doraIndicators || state.doras,
            state.winners.reduce((indicators, winner) =>
              winner.uraIndicators?.length > indicators.length ? winner.uraIndicators : indicators, [])));
          for (const winner of state.winners) {
            const section = document.createElement('section');
            const name = document.createElement('h3');
            name.className = 'kyoku-win-summary';
            const description = document.createElement('span');
            description.textContent = winText(winner, selfSeat, previousRound.kyoku)
              + ' · ' + winner.han + ' han ' + winner.fu + ' fu';
            const points = document.createElement('strong');
            points.className = 'kyoku-points';
            points.textContent = winPoints(winner, previousRound.kyoku);
            name.append(description, points);
            section.append(name);
            const hand = document.createElement('div');
            hand.className = 'kyoku-tiles';
            for (const tile of winner.hand) hand.append(tileElement(tile));
            const winning = tileElement(winner.tile);
            winning.classList.add('kyoku-winning');
            hand.append(winning);
            for (const meld of state.melds[winner.seat]) {
              const group = document.createElement('span');
              group.className = 'kyoku-meld';
              group.append(...meld.tiles.map(tile => tileElement(tile)));
              hand.append(group);
            }
            section.append(hand);
            const list = document.createElement('ul');
            for (const yaku of winner.yaku.filter(item => item.name !== "Ura Dora" || item.value > 0)) {
              const row = document.createElement('li');
              row.textContent = englishLabel(yaku.name) + ' · ' + yaku.value + (yaku.yakuman ? ' yakuman' : ' han');
              list.append(row);
            }
            section.append(list);
            popup.append(section);
          }
          if (result.type === 'ryuukyoku') {
            const outcome = document.createElement('h3');
            outcome.textContent = englishLabel(result.label);
            popup.append(outcome);
          }
          const table = document.createElement('table');
          table.innerHTML = '<thead><tr><th>Player</th><th>Settlement</th><th>Score</th></tr></thead>';
          const rows = document.createElement('tbody');
          for (let offset = 0; offset < 4; offset++) {
            const seat = (selfSeat + offset) % 4;
            const row = document.createElement('tr');
            for (const value of [['Self','Right','Across','Left'][offset],
              (state.scores[seat] - previousRound.scores[seat]).toLocaleString(undefined, {signDisplay:'exceptZero'}),
              state.scores[seat].toLocaleString()]) {
              const cell = document.createElement('td');
              cell.textContent = value;
              row.append(cell);
            }
            rows.append(row);
          }
          table.append(rows);
          popup.append(table);
        }
      }
      currentKey = key;
    }
    previousRound = round;
    previousLabel = label;
    if (closeOnDiscard && round.events.some(event => event.type === 'discard')) dismiss();
    if (popup) board.append(popup);
  };
}
