# Mahjong Soul review product

This directory owns replay review, the live read-only display, and their HTTP
server. Controller processes and their automation tests live in `../live/`.

Copy `.env.example` to `.env.local`, then run the product with `npm start`;
run its contract tests with `npm test`.
The server reads controller state through the local control API and never
starts, stops, or clicks a game client.
