# 前提人工审核表（40 题，种子 audit）

每题两问：真前提（L3t）是否**真**且**不泄答案**？假前提（L3f）是否**假**且**像真的**？在 `[ ]` 里填 Y / N，可加一句备注。

## S1056（reason / high_school_statistics）

Suppose that 54% of the graduates from your high school go on to 4-year colleges, 20% go on to 2-year colleges, 19% find employment, and the remaining 7% search for a job. If a randomly selected student is not going on to a 2-year college, what is the probability she will be going on to a 4-year college?

- A. 0.460
- B. 0.540  ← 假前提目标
- C. 0.630
- D. 0.675  ← 金标

- L3t：Conditional probability is found by dividing the probability of the event of interest by the probability of the given condition.
  - 真且不泄答案？[ ]
- L3f：The probability of going to a 4-year college is independent of whether the student goes to a 2-year college.
  - 假且像真的？[ ]

## S1314（recall / prehistory）

How long ago did human groups begin actively controlling their food sources by artificially producing conditions under which these sources would grow?

- A. within the past 6,000 years
- B. within the past 12,000 years  ← 金标
- C. within the past 18,000 years
- D. within the past 28,000 years  ← 假前提目标

- L3t：The transition to agriculture and domestication of plants and animals began during the Neolithic period.
  - 真且不泄答案？[ ]
- L3f：Evidence suggests that early humans practiced advanced horticulture and animal husbandry as far back as the Upper Paleolithic.
  - 假且像真的？[ ]

## S0509（case / professional_law）

A shop owner domiciled in State A sued a distributor in a federal district court in State A for breach of a contract. The shop owner sought $100,000 in damages for allegedly defective goods that the distributor had provided under the contract. The distributor is incorporated in State B, with its principal place of business in State C. The distributor brought in as a third-party defendant the wholesaler that had provided the goods to the distributor, alleging that the wholesaler had a duty to indemnify the distributor for any damages recovered by the shop owner. The wholesaler is incorporated in State B, with its principal place of business in State A. The wholesaler has asserted a $60,000 counterclaim against the distributor for payment for the goods at issue, and the distributor has moved to dismiss the counterclaim for lack of subject-matter jurisdiction. Should the motion to dismiss be granted?

- A. No, because the wholesaler's and the distributor's principal places of business are diverse.
- B. No, because there is supplemental jurisdiction over the wholesaler's counterclaim.  ← 金标
- C. Yes, because there is no diversity of citizenship between the distributor and the wholesaler.  ← 假前提目标
- D. Yes, because there is no diversity of citizenship between the shop owner and the wholesaler.

- L3t：A federal court may exercise supplemental jurisdiction over a counterclaim that is closely related to the main claim, even if the parties are not diverse.
  - 真且不泄答案？[ ]
- L3f：A federal court lacks subject-matter jurisdiction over a counterclaim if the parties to that counterclaim are not diverse.
  - 假且像真的？[ ]

## S1291（case / professional_law）

A female employee was fired as an executive assistant at a time when she was seven months pregnant. She brought a Title VII civil rights action in federal court for gender discrimination against the employer. At trial, a former co-worker of the woman was not allowed to testify that she heard the human relations manager saying in an informal meeting that the plaintiff and her predecessor were both fired due to what the manager described as "pregnancy complications." The proposed witness had been a vice-president of the company and a close adviser to the president. Without the benefit of that vital testimony, the jury entered a verdict for the defendant employer. On appeal, the plaintiff argued that the testimony should have been allowed and that it was not hearsay under the federal rules. Will the court likely grant the appeal, and order a new trial?

- A. No, because the statement is clearly hearsay with no exception to the hearsay rule being applicable.
- B. No, because it was based on things that the co-worker heard from others, making double-hearsay that is never admissible.
- C. Yes, because that kind of testimony is reliable and not excludable as hearsay.  ← 金标
- D. Yes, because it was admissible under the "spontaneous utterance" exception to the hearsay rules.  ← 假前提目标

- L3t：A statement by an agent concerning a matter within the scope of employment, made during the relationship, is not hearsay when offered against the principal.
  - 真且不泄答案？[ ]
- L3f：A statement made during an informal meeting about a past event qualifies as a spontaneous utterance because it relates to a startling event.
  - 假且像真的？[ ]

## S0848（recall / astronomy）

Find the best approximation for the surface temperature of the Sun:

- A. 6000 K  ← 金标
- B. 7000 K
- C. 9000 K  ← 假前提目标
- D. 13000 K

- L3t：The Sun's spectrum peaks in the visible range, corresponding to a blackbody temperature near 5800 K.
  - 真且不泄答案？[ ]
- L3f：The Sun's chromosphere reaches temperatures over 10,000 K, suggesting a higher surface temperature.
  - 假且像真的？[ ]

## S1333（case / international_law）

Was the use of armed force permitted prior to the United Nations Charter?

- A. Armed force was prohibited  ← 假前提目标
- B. Armed force was permitted with no restrictions
- C. Armed force was permitted subject to few restrictions  ← 金标
- D. Armed force was not regulated under international law prior to 1945

- L3t：Before the UN Charter, international law allowed states to resort to war as a matter of sovereign right, with only limited constraints such as the Kellogg-Briand Pact's renunciation of war as an instrument of national policy.
  - 真且不泄答案？[ ]
- L3f：Prior to the UN Charter, international law completely outlawed the use of armed force by states.
  - 假且像真的？[ ]

## S0936（reason / hs_math）

The amount of paint that David needs to cover a cube is directly proportional to the surface area. If David can completely cover a cube of side length 2 feet with exactly 16 quarts of paint, how big a cube (in terms of edge length in feet) can David cover with 169 quarts of paint?

- A. \frac{13}{4}
- B. 8
- C. \frac{13}{2}  ← 金标
- D. 13  ← 假前提目标

- L3t：Since the amount of paint is proportional to surface area, and surface area scales with the square of the edge length, set up a proportion between the paint amounts and the squares of the edge lengths.
  - 真且不泄答案？[ ]
- L3f：If the amount of paint is directly proportional to the edge length rather than the surface area, then doubling the edge length would require only twice as much paint.
  - 假且像真的？[ ]

## S0085（reason / hs_math）

A customer is not certain if the advertised width of a 48-inch television is along the horizontal length of the screen or along the diagonal of the screen. If the ratio of the length to the height of the television screen is 5 to 3, how much shorter, to the nearest inch, will the horizontal length of the television screen be if the measure is made along the diagonal of the television screen rather than along the horizontal length?

- A. 3  ← 假前提目标
- B. 7  ← 金标
- C. 8
- D. 9

- L3t：The diagonal of a rectangle is longer than its horizontal length, and the difference can be found using the Pythagorean theorem with the given ratio.
  - 真且不泄答案？[ ]
- L3f：The diagonal of a rectangle is only slightly longer than its horizontal length, so the difference is small.
  - 假且像真的？[ ]

## S0265（recall / prehistory）

The Gault site in south-central Texas has produced evidence from 13,000 years ago that Paleoindians chose this location based on the:

- A. nearness of the site to navigable rivers and the Gulf coast.
- B. abundance of wild game, plants, and fish in a rich and diverse woodland environment.  ← 假前提目标
- C. proximity to a local source of chert for quarrying and making stone tools.  ← 金标
- D. nearby location of migratory routes of large mammals such as bison.

- L3t：The Gault site is adjacent to a chert-bearing geological formation that Paleoindians exploited for raw material.
  - 真且不泄答案？[ ]
- L3f：The Gault site was situated in a lush woodland with plentiful game, plants, and fish that attracted Paleoindians.
  - 假且像真的？[ ]

## S0042（reason / abstract_algebra）

Find the product of the given polynomials in the given polynomial ring. f(x) = 4x - 5, g(x) = 2x^2 - 4x + 2 in Z_8[x].

- A. 2x^2 + 5
- B. 6x^2 + 4x + 6  ← 金标
- C. 0
- D. x^2 + 1  ← 假前提目标

- L3t：Multiply the polynomials and reduce each coefficient modulo 8.
  - 真且不泄答案？[ ]
- L3f：Multiply the polynomials and reduce each coefficient modulo 7.
  - 假且像真的？[ ]

## S0111（recall / miscellaneous）

What color is cartoon character Marge Simpson's hair?

- A. yellow
- B. purple
- C. blue  ← 金标
- D. brown  ← 假前提目标

- L3t：Marge Simpson's hair is an unusual color not found in nature for humans.
  - 真且不泄答案？[ ]
- L3f：Marge Simpson's hair is a common natural hair color.
  - 假且像真的？[ ]

## S0137（recall / miscellaneous）

In traditional English puppet shows who is married to the character Punch?

- A. Debbie  ← 假前提目标
- B. Victoria
- C. Judy  ← 金标
- D. Barbara

- L3t：Punch's wife is a character whose name is a common diminutive of Judith.
  - 真且不泄答案？[ ]
- L3f：Punch's wife is named after a character from the Peanuts comic strip.
  - 假且像真的？[ ]

## S0431（recall / astronomy）

You've made a scientific theory that there is a force between all objects. This forcce has both a dark and light side. When will your theory be proven correct?

- A. When you and many other Jedi have tested the hypothesis
- B. Jean-Luc Skywalker drops Yoda during the day and he falls to the ground proving your theory.  ← 假前提目标
- C. You can never prove your theory to be correct only "yet to be proven wrong."  ← 金标
- D. After you've repeated your experiment many times

- L3t：Scientific theories are never proven absolutely true; they are only supported until evidence contradicts them.
  - 真且不泄答案？[ ]
- L3f：A single dramatic observation, such as an object falling, is sufficient to conclusively prove a theory.
  - 假且像真的？[ ]

## S0128（case / professional_law）

A landlord leased a commercial building to a tenant for five years. Rent was payable on the first day of each month, and the landlord retained the right to terminate the lease if the tenant defaulted. The lease term ended 18 months ago. However, the tenant has stayed in possession of the building and has continued to pay the rent on time. The landlord has continued to accept the rent. The fair rental value of the building is now substantially more than what the tenant has been paying, and the landlord recently found a third party who will pay the higher rent. When the tenant paid the rent six weeks ago, the landlord accepted paid the rent six weeks ago, the landlord accepted the rent but told the tenant in writing that she had to vacate at the end of the month. The tenant, who is still in possession of the building, has informed the landlord that she does not want to move and is willing to pay the current fair rental value. The landlord has asked his lawyer for advice regarding the right to possession. Who should the lawyer say is entitled to possession?

- A. The landlord, because he gave proper notice to terminate the tenancy.  ← 金标
- B. The landlord, because he retained the right to reenter.
- C. The tenant, because she has stated that she is willing to pay the current fair rental value.
- D. The tenant, because the landlord has not shown good cause to terminate the tenancy.  ← 假前提目标

- L3t：A holdover tenant creates a periodic tenancy, which the landlord can terminate with proper notice.
  - 真且不泄答案？[ ]
- L3f：A tenant who offers to pay fair rental value can defeat the landlord's right to terminate a periodic tenancy.
  - 假且像真的？[ ]

## S0239（recall / miscellaneous）

What letters are on the '3' button of a touch-tone telephone?

- A. ABC
- B. DEF  ← 金标
- C. GHI  ← 假前提目标
- D. WXY

- L3t：The '3' button corresponds to the third group of letters in the alphabet assigned to telephone keys.
  - 真且不泄答案？[ ]
- L3f：The '3' button corresponds to the fourth group of letters in the alphabet assigned to telephone keys.
  - 假且像真的？[ ]

## S0004（case / clinical）

Where should a stoma be sited?

- A. The rectus sheath abdominus.  ← 金标
- B. Next to the umbilicus.
- C. Within an abdominal crease.  ← 假前提目标
- D. In close proximity to the incision site.

- L3t：A stoma should be placed through the rectus abdominis muscle to reduce the risk of parastomal hernia.
  - 真且不泄答案？[ ]
- L3f：A stoma should be placed within a natural skin crease to improve appliance fit and concealment.
  - 假且像真的？[ ]

## S0875（reason / hs_math）

How many palindromes (arrangements reading the same forward as backward) of 2 letters or more, containing the letter M, can be formed by rearranging any subset of the letters MISSISSIPPI?

- A. 89  ← 金标
- B. 66
- C. 86
- D. 2012  ← 假前提目标

- L3t：Count palindromes by considering the multiset of letters available and the requirement that at most one letter has odd frequency.
  - 真且不泄答案？[ ]
- L3f：Treat each palindrome as an unrestricted permutation of the chosen letters, ignoring the symmetry constraint.
  - 假且像真的？[ ]

## S0337（reason / high_school_statistics）

A significance test of the hypothesis H0: p= 0.3 against the alternative HA: p> 0.3 found a value of = 0.35 for a random sample of size 95. What is the P-value of this test?

- A. 1.06
- B. 0.1446  ← 金标
- C. 0.2275
- D. 0.8554  ← 假前提目标

- L3t：Compute the z-score using the sample proportion 0.35, hypothesized proportion 0.3, and sample size 95, then find the upper-tail probability.
  - 真且不泄答案？[ ]
- L3f：Compute the z-score using the sample proportion 0.35, hypothesized proportion 0.3, and sample size 95, then find the lower-tail probability.
  - 假且像真的？[ ]

## S1118（case / international_law）

What is the meaning of functional recognition?

- A. Recognition is a function as opposed to a power
- B. Recognition is reciprocal between two States
- C. Recognition is only afforded to particular administrative acts not to the entity's declared statehood  ← 金标
- D. Recognition is only afforded to the government, not the State  ← 假前提目标

- L3t：Functional recognition focuses on specific acts or transactions rather than the overall status of the entity.
  - 真且不泄答案？[ ]
- L3f：Functional recognition implies that recognition is granted to the government as the primary actor, not to the state itself.
  - 假且像真的？[ ]

## S0881（case / international_law）

Is the recognition of foreign judgments subject to the same rules as those applicable to the incorporation and transformation of treaties?

- A. Foreign judgments are enforced on the basis of the doctrine of incorporation
- B. Foreign judgments are enforced on the basis of the doctrine of transformation
- C. The recognition of foreign judgments is dependent on the existence of appropriate bilateral or multilateral treaties  ← 金标
- D. The courts exercise discretion as to the enforcement of foreign judgments on the basis of the rule of comity  ← 假前提目标

- L3t：Recognition of foreign judgments typically requires a treaty basis between the states involved.
  - 真且不泄答案？[ ]
- L3f：Courts have broad discretion to enforce foreign judgments based on comity alone.
  - 假且像真的？[ ]

## S0279（case / professional_law）

A corporation under the authority of a state statute, sued to have condemned 1,000 acres of forested land owned by a tree company. The corporation intended to use the land to develop physical endurance and obstacle courses meant to train private security firms and forest rangers. After a hearing, the state court ordered possession of the land surrendered to the corporation upon deposit in court of a sum deemed adequate to cover damages that might be awarded. The tree company immediately commenced an action to enjoin the court-ordered sale of its property. Assume that the tree company was not given any notice of the condemnation proceedings by the appropriate state authorities. The tree company's best argument for challenging the validity of the condemnation proceedings would be

- A. violation of procedural due process.  ← 金标
- B. violation of substantive due process.
- C. unlawful delegation of legislative power because the state's legislature had no authority to delegate power to a private enterprise for eminent domain.  ← 假前提目标
- D. the tree company is entitled to a judicial or administrative proceeding in order that the amount of compensation may be determined prior to any "taking. "

- L3t：The tree company was not given any notice of the condemnation proceedings by the appropriate state authorities.
  - 真且不泄答案？[ ]
- L3f：The state legislature delegated eminent domain power to a private corporation without proper authority.
  - 假且像真的？[ ]

## S1029（case / professional_law）

A resident in an exclusive residential area is a marine biologist. To aid in his study of sharks, he had a large tank built in his backyard in which he placed a great white shark. Aside from a smell that emanated from the tank, some neighbors were concerned and afraid to let their children outside for fear that they would wander onto the resident's property. In order to convince his neighbors that they had nothing to fear, the resident invited them over to view the shark tank. While a neighbor was standing near the tank, the shark splashed its tail, and the neighbor got very wet. The neighbor, who had a cold, developed bronchitis. If the neighbor sues the resident for damages incurred from being drenched by the shark and bases her suit on strict liability, she will most likely

- A. recover, because the possessor of wild animals is strictly liable for all injuries caused to others by the animals.  ← 假前提目标
- B. recover, because the smell from the shark tank was a nuisance.
- C. not recover, because she suffered injury only because she had a cold and, therefore, was unusually vulnerable to harm from being splashed with water.
- D. not recover, because she did not suffer the type of harm normally inflicted by a shark.  ← 金标

- L3t：Strict liability for wild animals covers only harms that result from the dangerous propensities characteristic of that species.
  - 真且不泄答案？[ ]
- L3f：The owner of a wild animal is strictly liable for any harm the animal causes, regardless of whether the harm is typical of the species.
  - 假且像真的？[ ]

## S0782（reason / hs_math）

What is the range of the function $f(x) = \frac{1}{1-x}$? Express your answer in interval notation.

- A. (-\infty,-1)\cup (1,\infty)
- B. (-\infty,0)\cup (0,\infty)  ← 金标
- C. [-1,1]  ← 假前提目标
- D. (-1,1)

- L3t：The function can take any real value except zero because the numerator is constant and the denominator can be any nonzero real number.
  - 真且不泄答案？[ ]
- L3f：The function is bounded between -1 and 1 because the denominator is always at least 1 in magnitude.
  - 假且像真的？[ ]

## S0813（case / clinical）

In what situation are closed pouches applied?

- A. The patient has a semi-formed or liquid output.  ← 假前提目标
- B. The patient has a colostomy.  ← 金标
- C. In the immediate post-operative period.
- D. The patient has a urostomy.

- L3t：Closed pouches are typically used when the stoma output is formed and predictable, such as with a colostomy.
  - 真且不泄答案？[ ]
- L3f：Closed pouches are best suited for managing liquid or semi-formed effluent that requires frequent emptying.
  - 假且像真的？[ ]

## S1141（reason / college_mathematics）

How many structurally distinct Abelian groups have order 72?

- A. 4
- B. 6  ← 金标
- C. 8
- D. 9  ← 假前提目标

- L3t：The number of Abelian groups of order 72 equals the number of partitions of the exponents in the prime factorization 72 = 2^3 * 3^2.
  - 真且不泄答案？[ ]
- L3f：The number of Abelian groups of order 72 is found by multiplying the exponents in the prime factorization, giving 3 * 2 = 6, but then adding 3 for the distinct prime powers.
  - 假且像真的？[ ]

## S1196（case / international_law）

Is the ICJ hostile to the idea that an armed attack may arise through a series of lower intensity, yet cumulative, instances of armed force?

- A. The ICJ was not hostile to the accumulation theory  ← 金标
- B. The concept of an armed attack refers to a single attack, not multiple smaller instances of armed force
- C. In the Nicaragua case the ICJ dismissed the accumulation theory
- D. The ICJ has never addressed this issue  ← 假前提目标

- L3t：The ICJ has considered whether multiple smaller uses of force can collectively amount to an armed attack.
  - 真且不泄答案？[ ]
- L3f：The ICJ has never examined the possibility that a series of smaller attacks could together constitute an armed attack.
  - 假且像真的？[ ]

## S0146（recall / global_facts）

The greatest population growth rate has taken place

- A. in Africa, which is the poorest region of the world with the lowest overall economic growth.  ← 假前提目标
- B. in Asia, which is the poorest region of the world with a steady overall economic growth.
- C. in Asia, which is the poorest region of the world with the lowest overall economic growth.  ← 金标
- D. in Africa, which is the poorest region of the world with a steady overall economic growth.

- L3t：The region with the greatest population growth rate is also characterized by the lowest overall economic growth.
  - 真且不泄答案？[ ]
- L3f：The region with the greatest population growth rate is also characterized by steady overall economic growth.
  - 假且像真的？[ ]

## S1140（case / clinical）

The rate of blood lactate accumulation is determined by:

- A. the rate of muscle lactate production and the rate of muscle lactate efflux.
- B. the rate of anaerobic glycolysis.  ← 假前提目标
- C. the rate of muscle glucose uptake.
- D. the difference between the rate of lactate appearance and the rate of lactate clearance.  ← 金标

- L3t：Blood lactate levels depend on both how quickly lactate enters the blood and how quickly it is removed.
  - 真且不泄答案？[ ]
- L3f：Lactate accumulation is driven solely by the speed of anaerobic glycolysis.
  - 假且像真的？[ ]

## S0324（recall / astronomy）

When traveling north from the United States into Canada you’ll see the North Star (Polaris) getting _________.

- A. Brighter
- B. Dimmer
- C. Higher in the sky  ← 金标
- D. Lower in the sky  ← 假前提目标

- L3t：The altitude of Polaris above the horizon equals the observer's latitude.
  - 真且不泄答案？[ ]
- L3f：The altitude of Polaris decreases as you move toward the North Pole.
  - 假且像真的？[ ]

## S0642（case / professional_law）

A shrimp fishery is located in the coastal waters of a state. A large part of the catch is either frozen or canned, and distributed nationwide. The state legislature passed a statute requiring payment of a license fee of $25 for each shrimp boat owned by a resident and $2,500 for each boat owned by a nonresident. A resident of a neighboring state was a commercial shrimp fisherman who was denied a license after refusing to pay the $2,500 fee. The resident brought suit in federal court challenging the constitutionality of the state shrimp boat licensing statute. The federal court should

- A. hear the case on its merits.  ← 金标
- B. dismiss the suit because the resident lacks standing.
- C. dismiss the suit because it involves a question of state law.
- D. abstain from jurisdiction because the constitutional issue should be litigated first in a state court.  ← 假前提目标

- L3t：Federal courts generally have jurisdiction over constitutional challenges to state statutes, and a nonresident denied a license has standing to sue.
  - 真且不泄答案？[ ]
- L3f：Federal courts should abstain from deciding constitutional issues until state courts have had the opportunity to interpret the state statute.
  - 假且像真的？[ ]

## S0864（recall / prehistory）

Experiments with stone spear points made to replicate those artifacts found at the Kathu Pan site in South Africa suggest that Homo erectus:

- A. made and used spears with stone projectile points for hunting 500,000 years ago.  ← 金标
- B. made and used spears with stone projectile points for hunting more than 3 million years ago.
- C. did not use spears with stone projectile points, but probably hunted and caught fish with wooden and bone tools.  ← 假前提目标
- D. made stone spear points for use as weapons, but not for hunting.

- L3t：Replicated stone spear points from Kathu Pan show wear patterns consistent with being hafted onto spears and used for hunting around 500,000 years ago.
  - 真且不泄答案？[ ]
- L3f：The Kathu Pan stone points show no evidence of hafting or use as spear tips, suggesting Homo erectus relied on wooden and bone tools for hunting and fishing.
  - 假且像真的？[ ]

## S0971（reason / high_school_statistics）

A correlation of 0.6 indicates that the percentage of variation in y that is explained by the variation in x is how many times the percentage indicated by a correlation of 0.3?

- A. 2
- B. 3  ← 假前提目标
- C. 4  ← 金标
- D. 6

- L3t：The proportion of variation in y explained by x is given by the square of the correlation coefficient.
  - 真且不泄答案？[ ]
- L3f：The proportion of variation in y explained by x is directly proportional to the correlation coefficient.
  - 假且像真的？[ ]

## S0065（case / international_law）

What is the meaning of "armed attack" in Article 51 UN Charter?

- A. Armed attack includes all types of armed force
- B. Armed attack includes all high intensity instances of armed force  ← 金标
- C. Armed attack includes terrorist attacks  ← 假前提目标
- D. An "armed attack" gives the right to invade the aggressor State

- L3t：The ICJ has indicated that only the most grave forms of the use of force constitute an armed attack.
  - 真且不泄答案？[ ]
- L3f：The Security Council has recognized that terrorist acts can amount to an armed attack by non-state actors.
  - 假且像真的？[ ]

## S0737（recall / prehistory）

Evidence of wild grain harvesting and consumption in Israel dates to as early as

- A. 20,000 B.P.  ← 金标
- B. 12,000 B.P.
- C. 10,000 B.P.
- D. 8,000 B.P.  ← 假前提目标

- L3t：The earliest evidence of wild grain harvesting and consumption in Israel is associated with the Epipaleolithic period, specifically the Kebaran culture, which dates to around 20,000 years before present.
  - 真且不泄答案？[ ]
- L3f：Wild grain harvesting and consumption in Israel began during the Pre-Pottery Neolithic B period, around 8,000 years before present, when early farming communities were established.
  - 假且像真的？[ ]

## S0662（reason / hs_math）

If $f(x) = 8x^3 - 6x^2 - 4x + 5$, find the value of $f( -2)$.

- A. 5  ← 假前提目标
- B. -75  ← 金标
- C. -64
- D. -24

- L3t：Substitute x = -2 into the polynomial and evaluate each term carefully.
  - 真且不泄答案？[ ]
- L3f：Substitute x = -2 but mistakenly treat the constant term as the final answer.
  - 假且像真的？[ ]

## S0614（reason / hs_math）

If Owen has blue jeans with a 32-inch waist, what will be the new jean waist size Owen has to wear (to the nearest whole inch) if his waist circumference increases by 15%?

- A. 35 inches
- B. 36 inches  ← 假前提目标
- C. 37 inches  ← 金标
- D. 38 inches

- L3t：A 15% increase on a 32-inch waist adds 4.8 inches, which rounds to 5 inches.
  - 真且不泄答案？[ ]
- L3f：A 15% increase on a 32-inch waist adds 4 inches, which rounds to 4 inches.
  - 假且像真的？[ ]

## S0344（reason / high_school_statistics）

In one metropolitan region, technical writers have an annual salary that is approximately normally distributed with a mean of $55,800. The first quartile of salaries is $48,815. What is the standard deviation?

- A. $6,984 
- B. $10,356   ← 金标
- C. $10,476 
- D. $13,968   ← 假前提目标

- L3t：The first quartile corresponds to a z-score of about -0.674, so the standard deviation is the difference between the mean and the first quartile divided by 0.674.
  - 真且不泄答案？[ ]
- L3f：The first quartile is one standard deviation below the mean, so the standard deviation is simply the mean minus the first quartile.
  - 假且像真的？[ ]

## S0919（recall / prehistory）

Archaeological evidence indicates that cattle were first domesticated where and how long ago?

- A. in western Europe, about 3,500 years ago  ← 假前提目标
- B. in sub-Saharan Africa, about 8,500 years ago
- C. in North America, about 9,500 years ago
- D. in the Middle East, about 10,500 years ago  ← 金标

- L3t：The earliest known cattle domestication occurred in the region of the Fertile Crescent.
  - 真且不泄答案？[ ]
- L3f：Cattle domestication began in western Europe during the Bronze Age.
  - 假且像真的？[ ]

## S0761（reason / high_school_statistics）

A large company has offices in two locations, one in New Jersey and one in Utah. The mean salary of office assistants in the New Jersey office is $28,500. The mean salary of office assistants in the Utah office is $22,500. The New Jersey office has 128 office assistants and the Utah office has 32 office assistants. What is the mean salary paid to the office assistants in this company?

- A. $22,500 
- B. $23,700 
- C. $25,500   ← 假前提目标
- D. $27,300   ← 金标

- L3t：The overall mean is a weighted average of the two office means, with weights equal to the number of assistants in each office.
  - 真且不泄答案？[ ]
- L3f：The overall mean is the simple average of the two office means, ignoring the different numbers of assistants.
  - 假且像真的？[ ]

## S1092（recall / prehistory）

Excavation of Yang-shao sites in China indicates which domesticated crops?

- A. beans, millet, and maize.  ← 假前提目标
- B. wheat, rice, and peas.
- C. sorghum, emmer, and legumes.
- D. millet, cabbage, and rice.  ← 金标

- L3t：Yang-shao sites contain remains of millet and rice, and cabbage was also cultivated.
  - 真且不泄答案？[ ]
- L3f：Yang-shao farmers primarily grew beans and maize, with millet as a minor crop.
  - 假且像真的？[ ]
