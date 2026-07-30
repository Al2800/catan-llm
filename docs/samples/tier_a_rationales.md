# Tier A rationale samples (ticket 10)

Generated from short expert games for human review.
All lines are POV-safe feature templates (SCOPE §7.4 / DATA_CONTRACT §8).

- `BUILD_SETTLEMENT` (random, BUILD_INITIAL_SETTLEMENT): settlement node=23; pips=9 (S+B+W); initial_placement
- `BUILD_ROAD` (random, BUILD_INITIAL_ROAD): road edge=(22,23); toward open node 22 pips=10 (S+B); longest_road threat=0
- `BUILD_SETTLEMENT` (valuefunction, BUILD_INITIAL_SETTLEMENT): settlement node=16; pips=12 (S+O+B); initial_placement
- `BUILD_ROAD` (valuefunction, BUILD_INITIAL_ROAD): road edge=(16,21); toward open node 21 pips=12 (O+B+S); longest_road threat=0
- `BUILD_SETTLEMENT` (weightedrandom, BUILD_INITIAL_SETTLEMENT): settlement node=33; pips=3 (W); port 3:1; initial_placement
- `BUILD_ROAD` (weightedrandom, BUILD_INITIAL_ROAD): road edge=(33,34); toward open node 34 pips=5 (W); longest_road threat=0
- `BUILD_SETTLEMENT` (alphabeta, BUILD_INITIAL_SETTLEMENT): settlement node=8; pips=11 (O+B+H); initial_placement
- `BUILD_ROAD` (alphabeta, BUILD_INITIAL_ROAD): road edge=(8,9); toward open node 9 pips=9 (O+H); longest_road threat=0
- `BUILD_SETTLEMENT` (alphabeta, BUILD_INITIAL_SETTLEMENT): settlement node=0; pips=9 (W+O+S); blocks BLUE,WHITE; initial_placement
- `BUILD_ROAD` (alphabeta, BUILD_INITIAL_ROAD): road edge=(0,20); toward open node 20 pips=9 (O+S); longest_road threat=0
- `BUILD_SETTLEMENT` (weightedrandom, BUILD_INITIAL_SETTLEMENT): settlement node=43; pips=8 (B+S); blocks WHITE; initial_placement
- `BUILD_ROAD` (weightedrandom, BUILD_INITIAL_ROAD): road edge=(43,44); toward open node 44 pips=3 (B); longest_road threat=0
- `BUILD_SETTLEMENT` (valuefunction, BUILD_INITIAL_SETTLEMENT): settlement node=11; pips=10 (H+O+W); blocks ORANGE; initial_placement
- `BUILD_ROAD` (valuefunction, BUILD_INITIAL_ROAD): road edge=(11,12); toward open node 12 pips=5 (H+W); longest_road threat=0
- `BUILD_SETTLEMENT` (random, BUILD_INITIAL_SETTLEMENT): settlement node=37; pips=3 (W+H); initial_placement
- `BUILD_ROAD` (random, BUILD_INITIAL_ROAD): road edge=(14,37); toward open node 14 pips=3 (W+H); longest_road threat=0
- `MOVE_ROBBER` (weightedrandom, MOVE_ROBBER): robber on pips=4 O; steal from RED (3 cards)
- `BUILD_ROAD` (weightedrandom, PLAY_TURN): road edge=(32,33); toward open node 32 pips=8 (O+W); longest_road threat=0
- `BUILD_CITY` (valuefunction, PLAY_TURN): city node=11; pips=10 (H+O+W); blocks ORANGE
- `BUILD_ROAD` (valuefunction, PLAY_TURN): road edge=(3,12); toward open node 3 pips=4 (W+H) / node 12 pips=5 (H+W); longest_road threat=1
- `MOVE_ROBBER` (weightedrandom, MOVE_ROBBER): robber on pips=2 S
- `MARITIME_TRADE` (weightedrandom, PLAY_TURN): maritime SSSDESERT→O; corrects hand[S=3,B=2]
- `MOVE_ROBBER` (weightedrandom, MOVE_ROBBER): robber on pips=1 W; steal from BLUE (4 cards)
- `BUILD_ROAD` (weightedrandom, PLAY_TURN): road edge=(40,44); toward open node 40 pips=7 (H+B) / node 44 pips=3 (B); longest_road threat=2
