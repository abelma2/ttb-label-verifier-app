# Example labels — try the app in a minute

Ready-to-use label images covering spirits, wine, and an import, so you can see the
verifier work without hunting for your own bottle photos.

| File(s) | What it is |
| --- | --- |
| `ABC.jpg` | synthetic **spirits** label (rye whisky) — front and back panels in one flat image: brand, class, ABV, net contents, address, government warning |
| `brand-label-new2.jpg` | synthetic **imported** liqueur label, one flat image — exercises the country-of-origin / imported-by case |
| `clear_baseline_3_Front.png` + `clear_baseline_3_Other.png` | synthetic **wine** label as a front + back pair — varietal, vintage, and appellation exercise the wine-only appellation check |
| `test_8_Front.jpeg` + `test_8_Other.jpeg` | real bottle **photos** (wine), front + back pair — what curved-glass, real-lighting input looks like |

**Quick demo (single mode):** upload `ABC.jpg` as the front label, leave the form blank,
and verify — that's the rules-only screening. For the paired sets, upload the `_Front`
file as the front and the `_Other` file as the back; both images are read together as
one label.

**Label-vs-application matching:** type values into the application form before
verifying (for `ABC.jpg`, try brand `ABC`, class `Straight Rye Whisky`, alcohol `45%
Alc/Vol`, net contents `750 mL`) — then change one value and re-run to watch the
mismatch get caught.

**Batch mode:** drop several files in together. Files pair into products by filename
stem (`test_8_Front` + `test_8_Other` become one product `test_8`), and each single
flat image is its own product.
