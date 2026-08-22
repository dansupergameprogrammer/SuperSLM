# T-2199 Phase E end-to-end confirmation

**Result: damped greedy works as an opt-in anti-loop decoder at the ruled operating point. Greedy remains the default.**

Operating point: `alpha=2`, `n=2`, `k=6`, `alpha_q15=65536`, `q=(493, 964, 487361)`.

## Full-generation production result

| Cell | Greedy locks | Damped locks | Reach (actual / B0) | Greedy rep-3 | Damped rep-3 | Paired delta +/- SE | Greedy cap/EOS | Damped cap/EOS | Effective ms/token ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 05_100 | 3 | 0 | 41/48 / 41/48 | 0.09063 | 0.00172 | -0.08890 +/- 0.02786 | 26/22 | 22/26 | 1.285x |
| 05_300 | 3 | 0 | 41/48 / 41/48 | 0.13959 | 0.00368 | -0.13591 +/- 0.03432 | 9/39 | 8/40 | 1.225x |
| 15_100 | 0 | 0 | 43/48 / 43/48 | 0.01873 | 0.00087 | -0.01786 +/- 0.00906 | 21/27 | 19/29 | 1.252x |
| 15_300 | 0 | 0 | 43/48 / 43/48 | 0.04139 | 0.00196 | -0.03943 +/- 0.01632 | 11/37 | 10/38 | 1.227x |

`cap/EOS` reports generations that reached the token ceiling / emitted a stop token. The effective wall-time ratio uses fresh paired greedy captures from the same executable and hardware, but includes model loading and different generated lengths, so it is an end-to-end observation, not an isolated selector-cost measurement. Phase D2a separately measures the wired selector against real forward cost.

Across the four fresh paired cells, damped/greedy wall time was 1.112x to 1.233x per 48-generation cell and 1.225x to 1.285x per generated token. This is a measured end-to-end harness cost, not attributed solely to selection. The final Phase D2a microbenchmark passed 29 checks per run. Three uncontended runs per real model put the complete selector at 0.3589%–0.3673% of forward cost on Qwen2.5 0.5B and 0.1221%–0.1305% on Qwen2.5 1.5B; the isolated renormalizer was 0.0002%–0.0007%. The two measurements have different scopes; Phase E does not claim that the microbenchmark explains the full end-to-end delta.

Production matched the B0 replay reach and locked-row ceiling in all four cells. The two 100-token cells were rerun after the ruling and matched their B0 token streams exactly: 0 mismatches across 96 generations. Greedy capture independently matched the frozen T-2193 baseline in all 192 model/length/prompt cases.

## Fidelity interpretation

The result supports the ruled product shape, not semantic equivalence. Damped greedy removes the measured loop locks and often improves prose, but it can alter facts, structure, count, formatting, and termination. In particular, `list_primed_00` demonstrates that legitimate repeated list syntax is penalized along with pathological repetition. Format-sensitive calls should use greedy or schema-constrained decoding unless the caller explicitly chooses this tradeoff.

## Real text, side by side (300-token cells)

The three known 0.5B lock cases come first, followed by fixed narration, flavor, and list cases. Text is decoded from the recorded token streams.

### 05_300

#### npc_dialogue_05

Prompt:

<pre>Write a line of dialogue for a rival tattoo artist trash-talking the player&#x27;s shop.</pre>

Greedy:

<pre>&quot;Ah, you&#x27;re finally here, you&#x27;re finally here! I&#x27;m the one who&#x27;s been here before, you know. But you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you</pre>

Damped greedy:

<pre>&quot;Ah, you&#x27;re finally here, and you&#x27;ve come to the shop. You&#x27;re looking for a tattoo? I&#x27;m glad you&#x27;re here. But first, let&#x27;s get this straight. What do you want your tattoo to say? To tell the truth, it&#x27;s all about you. So, what do ya want it to say?&quot;</pre>

#### npc_dialogue_08

Prompt:

<pre>Write a line of dialogue for the shop&#x27;s supply vendor haggling over the price of ink.</pre>

Greedy:

<pre>&quot;Ah, sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry. I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry. I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry</pre>

Damped greedy:

<pre>&quot;Ah, sorry about the price. I&#x27;m just a shopkeeper, not a vendor. Let me know if you need more ink or something else. Thank you for your patience.&quot;</pre>

#### npc_dialogue_11

Prompt:

<pre>Write a line of dialogue for the player&#x27;s mentor giving quiet encouragement after a hard day at the shop.</pre>

Greedy:

<pre>&quot;Remember, you&#x27;ve done it all, little one. You&#x27;ve done it all, and you&#x27;ve done it all. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for. Keep going, keep going, keep going,</pre>

Damped greedy:

<pre>&quot;Remember, you&#x27;ve done it all before. Keep going, little one. You&#x27;ve proven that you&#x27;re capable of more than you realize.&quot;</pre>

#### event_narration_00

Prompt:

<pre>Narrate the moment the tattoo shop opens for business on its very first morning.</pre>

Greedy:

<pre>In the bustling, neon-lit tattoo shop, the shopkeeper, a man named John, was just about to walk into the shop. The shop was bustling with activity, with the shopkeeper&#x27;s keen eyes scanning the area. He had a slight smile on his face, a sign of his enthusiasm. As he walked into the shop, he noticed a young woman, her face flushed with excitement. She was standing in the shop, her eyes fixed on the shop&#x27;s entrance. John, ever the businessman, took her hand and introduced himself. &quot;Good morning, Miss. Welcome to our shop. I&#x27;m John, the shopkeeper.&quot; The young woman smiled and walked over to the counter, where she placed a small order of coffee. John, ever the businessman, took her order and served her. He then noticed the shop&#x27;s new customer, a young man named David, walking in. He smiled and walked over to the counter, where he placed a small order of ink. John, ever the shopkeeper, took the order and began to prepare it. As he was preparing, he noticed the shop&#x27;s new customer, a young woman named Sarah, walking in. He smiled and introduced himself, &quot;Good morning, Sarah. I&#x27;m John, the shopkeeper. Welcome to our shop.&quot; The young woman smiled and walked over to the counter, where she placed a small order of ink. John, ever the shopkeeper, took the order and began to prepare it. He then noticed the</pre>

Damped greedy:

<pre>In the bustling, neon-lit tattoo shop, the shopkeeper, a man named John, was just about to walk into the room. The room was filled with the soft hum of a modern, high-tech tattoo machine. John&#x27;s eyes scanned the area, his keen senses picking up on the faint scent of ink and the distant hum from the other side of the shop. He approached the door, his hand resting on it, and stepped inside.</pre>

#### flavor_text_00

Prompt:

<pre>Write a short flavor-text description for an item called &quot;Steady Hands Energy Drink&quot; that appears in the game&#x27;s shop inventory.</pre>

Greedy:

<pre>In the heart of the bustling city, where the vibrant hues of the neon lights danced against the backdrop of towering skyscrapers, there was a unique item that captivated the hearts of adventurers and traders alike. It was a small, cylindrical vessel, its sleek design and smooth, metallic surface a striking contrast to the rough, rugged exterior of the city. This item was more than just a drink; it was a symbol of strength and power, a potent blend of the city&#x27;s energy, ensuring that those who drank it would not only survive but thrive in the city&#x27;s bustling environment. The drink was not just for the thirsty, but for the well-fed, for the city&#x27;s inhabitants who sought to sustain their strength and endurance. It was a tangible embodiment of the city&#x27;s spirit, a reminder of the city&#x27;s energy and the importance of it to those who lived in it.</pre>

Damped greedy:

<pre>In the heart of the bustling city, where the vibrant hues of red and gold danced across the streets, stood a shop that housed not just a store but a sanctuary for the weary traveler. This was the &quot;Tattoo Shop,&quot; where artists and enthusiasts alike sought to enhance their beauty with the power of tattoos.

In its backroom, the shop&#x27;s most prized item was a sleek, cylindrical container adorned with intricate patterns. The interior was adorned in vibrant colors, with shelves filled with vibrant, shimmering gemstones. Among these, a particular item stood out—a small, sleek bottle of what appeared to be a potent, amber-colored liquid.

The item&#x27;s name was &quot;Steady Hands Energy Drink.&quot; Its distinctive design was a blend of a traditional tattoo ink with a blendable, energizing essence. The drink was crafted to provide a temporary boost to the wearer&#x27;s energy, much like a quick burst of energy in a game.

This drink, though not a traditional one, was designed to be taken in pill form, which was why it was often sold in bulk. Its use was limited to its intended users, who sought it to rejuvenate their spirits and keep them energized. It was not meant to be consumed in any way, but rather consumed through the drink itself.

In the shop, customers would often be seen standing in front of the bottle, their faces a mix of excitement and perhaps a touch of nervousness. They would sip the drink, their eyes fixed on the</pre>

#### list_primed_00

Prompt:

<pre>Here is the shop&#x27;s current supply inventory:
Black Ink x12
Black Ink x12
Black Ink x12
Red Ink x8
Red Ink x8
Red Ink x8
Blue Ink x5
Blue Ink x5
Needle Cartridge x40
Needle Cartridge x40
Continue this inventory list with five more realistic entries.</pre>

Greedy:

<pre>Certainly! Here are five more realistic inventory entries for the tattoo shop:

1. **Gold Ink x10**
2. **Silver Ink x10**
3. **Black Ink x10**
4. **Red Ink x10**
5. **Blue Ink x10**
6. **Needle Cartridge x100**
7. **Black Ink x100**
8. **Red Ink x100**
9. **Blue Ink x100**
10. **Gold Ink x100**
11. **Silver Ink x100**
12. **Black Ink x100**
13. **Red Ink x100**
14. **Blue Ink x100**
15. **Gold Ink x100**
16. **Black Ink x100**
17. **Red Ink x100**
18. **Blue Ink x100**
19. **Gold Ink x100**
20. **Black Ink x100**
21. **Red Ink x100**
22. **Blue Ink x100**
23. **Gold Ink x100**
24. **Black Ink x100**
25. **Red Ink x100**
26. **Blue Ink x100**
27. **Gold Ink x100**
2</pre>

Damped greedy:

<pre>Certainly! Here are five more realistic inventory entries for the tattoo shop:

1. **Gold Ink x10**
2. Silver Ink ink x5
3. Bronze Ink x7
4. Platinum Ink x2
5. Diamond Ink x3

These entries would ensure variety and depth in the shop&#x27;s inventory, providing customers with a diverse range of ink options.</pre>

### 15_300

#### npc_dialogue_05

Prompt:

<pre>Write a line of dialogue for a rival tattoo artist trash-talking the player&#x27;s shop.</pre>

Greedy:

<pre>&quot;I&#x27;ll show you who&#x27;s the real master when I get my hands on that shop! You&#x27;ll be wishing you had a better connection to the gods to get out of this one!&quot;</pre>

Damped greedy:

<pre>&quot;I&#x27;ll show you who&#x27;s the real master when I get my hands on that shop! You&#x27;ll be wishing you had a better connection to the gods to get out of this one!&quot;</pre>

#### npc_dialogue_08

Prompt:

<pre>Write a line of dialogue for the shop&#x27;s supply vendor haggling over the price of ink.</pre>

Greedy:

<pre>&quot;Ah, ink! Such a fine commodity, isn&#x27;t it? Well, my dear customer, how much do you wish to part with for this ink? I&#x27;ve been working on this batch for quite some time, and it&#x27;s quite the treasure.&quot;</pre>

Damped greedy:

<pre>&quot;Ah, ink! Such a fine commodity. But let me tell you, my dear customer, this ink is not just any ordinary stuff. It&#x27;s crafted from the finest pigments, and it&#x27;s been through a rigorous process of refinement. I can assure you that it will last a lifetime, not fade or crack. Shall we discuss a price that&#x27;s both fair and respectful for your collection?&quot;</pre>

#### npc_dialogue_11

Prompt:

<pre>Write a line of dialogue for the player&#x27;s mentor giving quiet encouragement after a hard day at the shop.</pre>

Greedy:

<pre>&quot;Remember, sometimes the best kind of encouragement comes from a quiet whisper, isn&#x27;t it? Just knowing you&#x27;ve got my back, even when the world feels like it&#x27;s spinning out of control. Keep your hands steady, and your spirit strong. We&#x27;ll get through this, one stitch at a time.&quot;</pre>

Damped greedy:

<pre>&quot;Remember, sometimes the best kind of encouragement comes from a quiet whisper in the ear. Keep your spirits high, and you&#x27;ll see the world through new eyes.&quot;</pre>

#### event_narration_00

Prompt:

<pre>Narrate the moment the tattoo shop opens for business on its very first morning.</pre>

Greedy:

<pre>The sun rises on the quaint town, casting a golden glow over the streets. The air is crisp and fresh, and the first rays of sunlight filter through the windows, illuminating the bustling scene outside. The aroma of fresh coffee and the tang of early morning dew fills the air, promising a day of new beginnings.

The door creaks open, and a soft murmur of excited chatter fills the air. A group of young apprentices, each with a tattoo in mind, rush in, their eyes wide with excitement. They rush to the back, where the master tattooist, a gentle and wise woman named Elara, is already hard at work.

&quot;Good morning, everyone!&quot; Elara&#x27;s voice crackles through the door, her tone filled with warmth and welcome. &quot;We&#x27;re just getting started, but I can already feel the energy in the air. It&#x27;s going to be a busy day!&quot;

The apprentices cheer, their hands already in the air, ready to dive into their work. They rush to the waiting clients, each one eager to have their own unique piece of art. The first client, a young artist named Lila, is already in the waiting room, her eyes wide with wonder.

&quot;Can I have a tattoo of a dragon?&quot; she asks, her voice dripping with excitement. &quot;I want it to be big and fierce!&quot;

Elara smiles, her eyes twinkling with joy. &quot;Of course, Lila. Dragons are a powerful symbol of courage and strength</pre>

Damped greedy:

<pre>The sun rises on the quaint town, casting a golden glow over the streets. The air is crisp and fresh, and the first rays of sunlight filter through the windows, illuminating the bustling scene outside. A lone figure stands at the entrance of the tattoo shop, their eyes wide with excitement and anticipation.

The door creaks open, and a soft murmur of footsteps echoes through to the interior. The proprietor, a man with a kind smile and a steady hand, steps out, holding a steaming cup of coffee. He waves his hand in greeting, his voice warm and friendly.

&quot;Good morning, folks! Welcome to our first day of business! I&#x27;m glad to have you all here today. We&#x27;ve got some amazing artists and some incredible designs in store for you. Let&#x27;s see what we can come up with this morning!&quot;

As he speaks, he gestures to his side, where a small cart is set up, filled with all sorts of ink and tools. It&#x27;s a bustling little scene, with customers and artists alike coming and going, each one eager to get their hands on whatever new designs are being created.

Behind the counter, the shop&#x27;s first tattoo artist is already hard at work, sketching designs on a notepad. Another artist sits at a small table,sembling up a new piece,, her hands moving with practiced ease as she works her magic.

As the morning goes on,,,, the shop fills up like a magnet, with people from all</pre>

#### flavor_text_00

Prompt:

<pre>Write a short flavor-text description for an item called &quot;Steady Hands Energy Drink&quot; that appears in the game&#x27;s shop inventory.</pre>

Greedy:

<pre>Ah, indeed, here we are with a most peculiar concoction, the &quot;Steady Hands Energy Drink.&quot; Crafted by the the very best of the tattoo shop&#x27;s master craftsmen, this elixir is not just a drink, but a testament to the skill and dedication of our artisans. It&#x27;s said to imbue the drinker with a newfound sense of control and focus, perfect for those who find themselves in the midst of their most intricate and challenging projects. Whether you&#x27;re a seasoned artist or just beginning, this drink is a must-have for anyone who values their craft and their ability to hold their own.</pre>

Damped greedy:

<pre>Ah, indeed! Let me tell you about this potent concoction, crafted by the finest mixologists in the land. The &quot;Steady Hands Energy Drink&quot; is more than just a refreshment; it&#x27;s a potion of endurance and focus, designed to keep even the most ardent of tattoo artists at the top of their game. Sip from this elix, and you&#x27;ll find yourself afloat in a sea of ink, ready to tackle any challenge that comes your way.</pre>

#### list_primed_00

Prompt:

<pre>Here is the shop&#x27;s current supply inventory:
Black Ink x12
Black Ink x12
Black Ink x12
Red Ink x8
Red Ink x8
Red Ink x8
Blue Ink x5
Blue Ink x5
Needle Cartridge x40
Needle Cartridge x40
Continue this inventory list with five more realistic entries.</pre>

Greedy:

<pre>Certainly, here are five more realistic entries to add to the inventory list:

- Tattoo Kit x1
- Razor Blade x2
- Cotton Balls x10
- Ink Wash x2
- Sponge x10
- Ink Brushes x5
- Tattoo Paper x10
- Wax x2
- Band-Aids x10
- Tweezers x2
- Razor Blades x2
- Cotton Balls x10
- Ink Wash x2
- Sponge x10
- Ink Brushes x5
- Tattoo Paper x10
- Wax x2
- Band-Aids x10
- Tweezers x2
- Tattoo Ink x1
- Ink Brushes x5
- Tattoo Paper x10
- Wax x2
- Band-Aids x10
- Tweezers x2
- Tattoo Ink x1
- Razor Blades x2
- Cotton Balls x10
- Ink Wash x2
- Sponge x10
- Ink Brushes x5
- Tattoo Paper x10
- Razor Blades x2
- Cotton Balls x10
- Ink Wash x2
- Sponge x10
- Ink Brushes x5
- Wax x2
- Band-Aids x10
- Tweezers x2
- Ink Inks x1
- Ink Brushes x5
- Tattoo Paper x10
- Wax x2
- Band-Aids</pre>

Damped greedy:

<pre>Certainly, here are five more realistic entries to add to the inventory list:

- Tattoo Kit x1
- Razor Blade x2
  - Includes 50 blades
- Ink Pouch x5
    - Each can hold up to 10ml of ink
- Cotton Swabs x30
- Clean Cloth x15
- Sterile Gloves x20
- Ink Bottle x4
  - Holds 2ml each of Black, Red, and Blue inks
  - Also contains a small amount of clear ink for touch-up
- Needle Holder x8
  - Each holds 12 needles
- Tattoo Wax x6
   - Used for skin preparation and tattoo application
- Waxing Kit (Included in the kit) x1
     - Includes a waxing wand and a set of wax strips
- Tattoo Sponges x5
     These come in different sizes and are used for various tattooing techniques
- Tattoo Sponge Holder (included in the kit)
- Tattoo Brush x1 (used for blending and smoothing out the ink)

This expanded inventory should provide a more comprehensive and realistic experience for the tattooist and customer.</pre>
