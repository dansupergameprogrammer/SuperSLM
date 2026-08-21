# T-2199 Phase B0 decision packet

This packet reports calibration evidence; it does not automatically select a shipping row.

## Primary-grid interpretation

The production-selector replay reached its maximum at alpha=3, n=2/3, k=6/10 (170/192 cells). Alpha=2, n=2/3, k=6/10 reached 168/192. At k=6, alpha=2 retains more component-A governance margin than alpha=3; k=10 adds cost without reach at either strength. The two orders tie on replay reach, so n=2 is the simpler row to weigh.

Selected-for-confirmation replay row: `{"05_100_locked_moved": 3, "05_100_reach": 41, "05_100_rep3_baseline": 0.09062516565579103, "05_100_rep3_baseline_without_locks": 0.04564643520291179, "05_100_rep3_ceiling": 0.0, "05_100_rep3_ceiling_without_locks": 0.0, "05_100_rep3_drop_over_se": 3.2465599829882477, "05_100_rep3_drop_over_se_without_locks": 3.892281909109055, "05_100_rep3_se": 0.027914212622178768, "05_100_rep3_se_without_locks": 0.011727422696718358, "05_300_locked_moved": 3, "05_300_reach": 41, "05_300_rep3_baseline": 0.13958775324289352, "05_300_rep3_baseline_without_locks": 0.08737235066266585, "05_300_rep3_ceiling": -2.7755575615628914e-17, "05_300_rep3_ceiling_without_locks": 0.0, "05_300_rep3_drop_over_se": 4.075770137182183, "05_300_rep3_drop_over_se_without_locks": 4.714761914456647, "05_300_rep3_se": 0.034248190782269844, "05_300_rep3_se_without_locks": 0.018531657005788613, "15_100_locked_moved": 0, "15_100_reach": 43, "15_100_rep3_baseline": 0.018730548899721835, "15_100_rep3_baseline_without_locks": 0.018730548899721835, "15_100_rep3_ceiling": 0.0, "15_100_rep3_ceiling_without_locks": 0.0, "15_100_rep3_drop_over_se": 1.9700859403676718, "15_100_rep3_drop_over_se_without_locks": 1.9700859403676718, "15_100_rep3_se": 0.009507478083025253, "15_100_rep3_se_without_locks": 0.009507478083025253, "15_300_locked_moved": 0, "15_300_reach": 43, "15_300_rep3_baseline": 0.041387233810933545, "15_300_rep3_baseline_without_locks": 0.041387233810933545, "15_300_rep3_ceiling": 0.0, "15_300_rep3_ceiling_without_locks": 0.0, "15_300_rep3_drop_over_se": 2.409127039752252, "15_300_rep3_drop_over_se_without_locks": 2.409127039752252, "15_300_rep3_se": 0.017179348837988093, "15_300_rep3_se_without_locks": 0.017179348837988093, "all_component_b": true, "alpha": 2.0, "k": 6, "max_generations_over_10x": 30, "median_cost_ns": 330.20500000000004, "min_margin_a": 2.3752830188679246, "n": 2, "worst_stability_max": 150.088, "worst_stability_median": 14.9519}`

Higher-strength challenger replay row: `{"05_100_locked_moved": 3, "05_100_reach": 41, "05_100_rep3_baseline": 0.09062516565579103, "05_100_rep3_baseline_without_locks": 0.04564643520291179, "05_100_rep3_ceiling": 0.0, "05_100_rep3_ceiling_without_locks": 0.0, "05_100_rep3_drop_over_se": 3.2465599829882477, "05_100_rep3_drop_over_se_without_locks": 3.892281909109055, "05_100_rep3_se": 0.027914212622178768, "05_100_rep3_se_without_locks": 0.011727422696718358, "05_300_locked_moved": 3, "05_300_reach": 41, "05_300_rep3_baseline": 0.13958775324289352, "05_300_rep3_baseline_without_locks": 0.08737235066266585, "05_300_rep3_ceiling": -2.7755575615628914e-17, "05_300_rep3_ceiling_without_locks": 0.0, "05_300_rep3_drop_over_se": 4.075770137182183, "05_300_rep3_drop_over_se_without_locks": 4.714761914456647, "05_300_rep3_se": 0.034248190782269844, "05_300_rep3_se_without_locks": 0.018531657005788613, "15_100_locked_moved": 0, "15_100_reach": 44, "15_100_rep3_baseline": 0.018730548899721835, "15_100_rep3_baseline_without_locks": 0.018730548899721835, "15_100_rep3_ceiling": 0.0, "15_100_rep3_ceiling_without_locks": 0.0, "15_100_rep3_drop_over_se": 1.9700859403676718, "15_100_rep3_drop_over_se_without_locks": 1.9700859403676718, "15_100_rep3_se": 0.009507478083025253, "15_100_rep3_se_without_locks": 0.009507478083025253, "15_300_locked_moved": 0, "15_300_reach": 44, "15_300_rep3_baseline": 0.041387233810933545, "15_300_rep3_baseline_without_locks": 0.041387233810933545, "15_300_rep3_ceiling": 0.0, "15_300_rep3_ceiling_without_locks": 0.0, "15_300_rep3_drop_over_se": 2.409127039752252, "15_300_rep3_drop_over_se_without_locks": 2.409127039752252, "15_300_rep3_se": 0.017179348837988093, "15_300_rep3_se_without_locks": 0.017179348837988093, "all_component_b": true, "alpha": 3.0, "k": 6, "max_generations_over_10x": 30, "median_cost_ns": 330.20500000000004, "min_margin_a": 1.5835220125786162, "n": 2, "worst_stability_max": 150.088, "worst_stability_median": 14.9519}`

All commissioning controls passed: alpha=0 identity, alpha=30 vitality, synthetic component-A nonzero margin, and synthetic component-B rejection.

## Genuine autoregressive rollout (48 prompts, max-new=100)

| Model / decoder | Reach | Locks | Capped | EOS | Mean tokens | Mean rep-3 | Paired rep-3 delta +/- SE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5B greedy | -- | 3 | 26 | 22 | 71.83 | 0.09063 | -- |
| 0.5B alpha=2 n=2 k=6 | 41/48 | 0 | 22 | 26 | 68.42 | 0.00172 | -0.08890 +/- 0.02786 |
| 0.5B alpha=3 n=2 k=6 | 41/48 | 0 | 24 | 24 | 68.96 | 0.00133 | -0.08929 +/- 0.02784 |
| 1.5B greedy | -- | 0 | 21 | 27 | 68.06 | 0.01873 | -- |
| 1.5B alpha=2 n=2 k=6 | 43/48 | 0 | 19 | 29 | 67.04 | 0.00087 | -0.01786 +/- 0.00906 |

The counterfactual replay is a reach/lock ceiling. The table above is the achieved result after every divergence is fed back through the model.

## Human-readable sample

The three 0.5B baseline loop-lock prompts are included first, followed by one fixed prompt from each remaining corpus category. Text is decoded from recorded token IDs.

### 0.5B

#### npc_dialogue_05

Greedy:

<pre>&quot;Ah, you&#x27;re finally here, you&#x27;re finally here! I&#x27;m the one who&#x27;s been here before, you know. But you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you&#x27;re here, you</pre>

Damped greedy (alpha=2, n=2, k=6):

<pre>&quot;Ah, you&#x27;re finally here, and you&#x27;ve come to the shop. You&#x27;re looking for a tattoo? I&#x27;m glad you&#x27;re here. But first, let&#x27;s get this straight. What do you want your tattoo to say? To tell the truth, it&#x27;s all about you. So, what do ya want it to say?&quot;</pre>

#### npc_dialogue_08

Greedy:

<pre>&quot;Ah, sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry. I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry. I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry, I&#x27;m sorry</pre>

Damped greedy (alpha=2, n=2, k=6):

<pre>&quot;Ah, sorry about the price. I&#x27;m just a shopkeeper, not a vendor. Let me know if you need more ink or something else. Thank you for your patience.&quot;</pre>

#### npc_dialogue_11

Greedy:

<pre>&quot;Remember, you&#x27;ve done it all, little one. You&#x27;ve done it all, and you&#x27;ve done it all. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for. Keep going, keep going, keep going, and you&#x27;ll find what you&#x27;re looking for</pre>

Damped greedy (alpha=2, n=2, k=6):

<pre>&quot;Remember, you&#x27;ve done it all before. Keep going, little one. You&#x27;ve proven that you&#x27;re capable of more than you realize.&quot;</pre>

#### event_narration_00

Greedy:

<pre>In the bustling, neon-lit tattoo shop, the shopkeeper, a man named John, was just about to walk into the shop. The shop was bustling with activity, with the shopkeeper&#x27;s keen eyes scanning the area. He had a slight smile on his face, a sign of his enthusiasm. As he walked into the shop, he noticed a young woman, her face flushed with excitement. She was standing in the shop, her eyes fixed on the shop&#x27;s entrance. John, ever the</pre>

Damped greedy (alpha=2, n=2, k=6):

<pre>In the bustling, neon-lit tattoo shop, the shopkeeper, a man named John, was just about to walk into the room. The room was filled with the soft hum of a modern, high-tech tattoo machine. John&#x27;s eyes scanned the area, his keen senses picking up on the faint scent of ink and the distant hum from the other side of the shop. He approached the door, his hand resting on it, and stepped inside.</pre>

#### flavor_text_00

Greedy:

<pre>In the heart of the bustling city, where the vibrant hues of the neon lights danced against the backdrop of towering skyscrapers, there was a unique item that captivated the hearts of adventurers and traders alike. It was a small, cylindrical vessel, its sleek design and smooth, metallic surface a striking contrast to the rough, rugged exterior of the city. This item was more than just a drink; it was a symbol of strength and power, a potent blend of the city&#x27;s energy, ensuring that</pre>

Damped greedy (alpha=2, n=2, k=6):

<pre>In the heart of the bustling city, where the vibrant hues of red and gold danced across the streets, stood a shop that housed not just a store but a sanctuary for the weary traveler. This was the &quot;Tattoo Shop,&quot; where artists and enthusiasts alike sought to enhance their beauty with the power of tattoos.

In its backroom, the shop&#x27;s most prized item was a sleek, cylindrical container adorned with intricate patterns. The interior was adorned in vibrant colors, with shelves filled with vibrant,</pre>

#### list_primed_00

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
9. **Blue Ink x100</pre>

Damped greedy (alpha=2, n=2, k=6):

<pre>Certainly! Here are five more realistic inventory entries for the tattoo shop:

1. **Gold Ink x10**
2. Silver Ink ink x5
3. Bronze Ink x7
4. Platinum Ink x2
5. Diamond Ink x3

These entries would ensure variety and depth in the shop&#x27;s inventory, providing customers with a diverse range of ink options.</pre>

### 1.5B

#### npc_dialogue_05

Greedy:

<pre>&quot;I&#x27;ll show you who&#x27;s the real master when I get my hands on that shop! You&#x27;ll be wishing you had a better connection to the gods to get out of this one!&quot;</pre>

Damped greedy (alpha=2, n=2, k=6):

<pre>&quot;I&#x27;ll show you who&#x27;s the real master when I get my hands on that shop! You&#x27;ll be wishing you had a better connection to the gods to get out of this one!&quot;</pre>

#### npc_dialogue_08

Greedy:

<pre>&quot;Ah, ink! Such a fine commodity, isn&#x27;t it? Well, my dear customer, how much do you wish to part with for this ink? I&#x27;ve been working on this batch for quite some time, and it&#x27;s quite the treasure.&quot;</pre>

Damped greedy (alpha=2, n=2, k=6):

<pre>&quot;Ah, ink! Such a fine commodity. But let me tell you, my dear customer, this ink is not just any ordinary stuff. It&#x27;s crafted from the finest pigments, and it&#x27;s been through a rigorous process of refinement. I can assure you that it will last a lifetime, not fade or crack. Shall we discuss a price that&#x27;s both fair and respectful for your collection?&quot;</pre>

#### npc_dialogue_11

Greedy:

<pre>&quot;Remember, sometimes the best kind of encouragement comes from a quiet whisper, isn&#x27;t it? Just knowing you&#x27;ve got my back, even when the world feels like it&#x27;s spinning out of control. Keep your hands steady, and your spirit strong. We&#x27;ll get through this, one stitch at a time.&quot;</pre>

Damped greedy (alpha=2, n=2, k=6):

<pre>&quot;Remember, sometimes the best kind of encouragement comes from a quiet whisper in the ear. Keep your spirits high, and you&#x27;ll see the world through new eyes.&quot;</pre>

#### event_narration_00

Greedy:

<pre>The sun rises on the quaint town, casting a golden glow over the streets. The air is crisp and fresh, and the first rays of sunlight filter through the windows, illuminating the bustling scene outside. The aroma of fresh coffee and the tang of early morning dew fills the air, promising a day of new beginnings.

The door creaks open, and a soft murmur of excited chatter fills the air. A group of young apprentices, each with a tattoo in mind, rush in, their</pre>

Damped greedy (alpha=2, n=2, k=6):

<pre>The sun rises on the quaint town, casting a golden glow over the streets. The air is crisp and fresh, and the first rays of sunlight filter through the windows, illuminating the bustling scene outside. A lone figure stands at the entrance of the tattoo shop, their eyes wide with excitement and anticipation.

The door creaks open, and a soft murmur of footsteps echoes through to the interior. The proprietor, a man with a kind smile and a steady hand, steps out, holding a</pre>

#### flavor_text_00

Greedy:

<pre>Ah, indeed, here we are with a most peculiar concoction, the &quot;Steady Hands Energy Drink.&quot; Crafted by the the very best of the tattoo shop&#x27;s master craftsmen, this elixir is not just a drink, but a testament to the skill and dedication of our artisans. It&#x27;s said to imbue the drinker with a newfound sense of control and focus, perfect for those who find themselves in the midst of their most intricate and challenging projects. Whether you&#x27;re a</pre>

Damped greedy (alpha=2, n=2, k=6):

<pre>Ah, indeed! Let me tell you about this potent concoction, crafted by the finest mixologists in the land. The &quot;Steady Hands Energy Drink&quot; is more than just a refreshment; it&#x27;s a potion of endurance and focus, designed to keep even the most ardent of tattoo artists at the top of their game. Sip from this elix, and you&#x27;ll find yourself afloat in a sea of ink, ready to tackle any challenge that comes your way.</pre>

#### list_primed_00

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
- Sponge</pre>

Damped greedy (alpha=2, n=2, k=6):

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
  - Holds 2ml each of Black, Red, and Blue in</pre>
