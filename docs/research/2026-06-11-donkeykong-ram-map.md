# Research notes: Atari 2600 Donkey Kong RAM map (for the scripted demonstrator)

One-line summary: verified getRAM() byte indices for Mario position, climbing
state, barrels, and the barrels-screen ladder layout, cross-checked between
OCAtari and the Dennis Debro disassembly.

Convention: `ale.getRAM()[i]` is zero-page address `0x80 + i` (ALE
stella_environment.cpp). All values below are raw RAM units, not OCAtari
screen pixels. Mario y grows DOWNWARD (top platform 12, bottom about 154).

## Key bytes

| getRAM idx | meaning | notes |
|---|---|---|
| 19 | Mario x | |
| 27 | Mario y | raw; OCAtari screen y = raw + 22 |
| 20 | marioDirection | 0xFF climbing up, 0xFE climbing down, bit 7 clear on floor |
| 5 | jumpHangTime | nonzero while jumping |
| 6 | hammerTime | nonzero while holding hammer (cannot climb) |
| 12 | gameState | bit 1 = LEVEL_COMPLETED, bit 0 = START_NEW_LEVEL |
| 16 | currentGameBoard | 0 = barrels screen, 1 = rivets screen |
| 35 | lives | |
| 36 | bonusTimer | OCAtari wrongly calls this Score |
| 7-8 | playerScore | 2 bytes BCD, display adds two zeros |
| 0-3 | barrel y (4 slots) | 0 = slot empty |
| 65-68 | barrel x (4 slots) | |
| 73-76 | barrelLadderNumber | which ladder a barrel descends |
| 106 | ladderNumber | ladder Mario is on |
| 109 | marioPlatform | 0 = bottom, 5 = top (barrels screen) |

## Mario-usable ladders, barrels screen (raw units)

(x, bottom y, top y); press UP within x tolerance 0..2 at bottom y:

(109, 154, 132), (81, 130, 101), (49, 127, 104), (89, 101, 74),
(109, 99, 76), (69, 73, 46), (49, 71, 48), (109, 43, 21), (77, 21, 5)

The last (x=77, 21 to 5) is ascend-only and leads to the goal. Walk limits
about x in [41, 117]. Climbing chain used by the demonstrator:
(109,154) -> (81,130) -> (89,101) -> (69,73) -> (109,43) -> (77,21).

Full details and source quotes in the research agent report; primary sources:
- https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/donkeykong.py
- https://github.com/milnak/atari-vcs-disassembly/blob/master/Third%20Party/Donkey%20Kong%20(1982)%20(Coleco)%20-%20Dennis%20Debro.asm
- https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/environment/stella_environment.cpp
