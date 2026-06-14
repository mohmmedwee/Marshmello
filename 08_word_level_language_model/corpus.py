"""
Build the embedded toy corpus (public-domain-style prose).
Kept separate from train.py so the training script stays readable.
"""

# Each passage is a few hundred words of simplified Shakespeare-adjacent prose.
PASSAGES: list[str] = [
    """
    To be, or not to be, that is the question. Whether it is nobler in the mind
    to suffer the slings and arrows of outrageous fortune, or to take arms against
    a sea of troubles and by opposing end them. To die, to sleep, no more, and by
    a sleep to say we end the heart-ache and the thousand natural shocks that flesh
    is heir to. It is a consummation devoutly to be wished. To die, to sleep, to
    sleep perchance to dream. Ay, there is the rub, for in that sleep of death what
    dreams may come when we have shuffled off this mortal coil must give us pause.
    There is the respect that makes calamity of so long life. For who would bear
    the whips and scorns of time, the oppressor's wrong, the proud man's contumely,
    the pangs of despised love, the law's delay, the insolence of office, and the
    spurns that patient merit of the unworthy takes, when he himself might his
    quietus make with a bare bodkin. Who would fardels bear, to grunt and sweat
    under a weary life, but that the dread of something after death, the undiscovered
    country from whose bourn no traveller returns, puzzles the will and makes us
    rather bear those ills we have than fly to others that we know not of. Thus
    conscience does make cowards of us all, and thus the native hue of resolution
    is sicklied over with the pale cast of thought, and enterprises of great pitch
    and moment with this regard their currents turn awry and lose the name of action.
    """,
    """
    All the world's a stage, and all the men and women merely players. They have
    their exits and their entrances, and one man in his time plays many parts,
    his acts being seven ages. At first the infant, mewling and puking in the nurse's
    arms. Then the whining school-boy with his satchel and shining morning face,
    creeping like snail unwillingly to school. And then the lover, sighing like
    furnace, with a woeful ballad made to his mistress' eyebrow. Then a soldier,
    full of strange oaths and bearded like the pard, jealous in honour, sudden and
    quick in quarrel, seeking the bubble reputation even in the cannon's mouth.
    And then the justice, in fair round belly with good capon lined, with eyes severe
    and beard of formal cut, full of wise saws and modern instances. The sixth age
    shifts into the lean and slippered pantaloon, with spectacles on nose and pouch
    on side, his youthful hose well saved a world too wide for his shrunk shank,
    and his big manly voice turning again toward childish treble, pipes and whistles
    in his sound. Last scene of all, that ends this strange eventful history, is
    second childishness and mere oblivion, sans teeth, sans eyes, sans taste, sans
    everything. Such is the common track of human life from cradle unto grave.
    """,
    """
    Friends, Romans, countrymen, lend me your ears. I come to bury Caesar, not to
    praise him. The evil that men do lives after them, the good is oft interred
    with their bones. So let it be with Caesar. The noble Brutus hath told you
    Caesar was ambitious. If it were so, it was a grievous fault, and grievously
    hath Caesar answered it. Here, under leave of Brutus and the rest, for Brutus
    is an honourable man, so are they all, all honourable men, come I to speak in
    Caesar's funeral. He was my friend, faithful and just to me, but Brutus says
    he was ambitious, and Brutus is an honourable man. He hath brought many captives
    home to Rome whose ransoms did the general coffers fill. Did this in Caesar
    seem ambitious? When that the poor have cried, Caesar hath wept. Ambition should
    be made of sterner stuff. Yet Brutus says he was ambitious, and Brutus is an
    honourable man. You all did see that on the Lupercal I thrice presented him
    a kingly crown, which he did thrice refuse. Was this ambition? Yet Brutus says
    he was ambitious, and sure he is an honourable man. I speak not to disprove
    what Brutus spoke, but here I am to speak what I do know. You all did love
    him once, not without cause. What cause withholds you then to mourn for him?
    O judgment, thou art fled to brutish beasts, and men have lost their reason.
    """,
    """
    Double, double toil and trouble, fire burn and caldron bubble. By the pricking
    of my thumbs, something wicked this way comes. Open locks, whoever knocks.
    Fair is foul, and foul is fair, hover through the fog and filthy air. When shall
    we three meet again in thunder, lightning, or in rain? When the hurlyburly's
    done, when the battle's lost and won. That will be ere the set of sun. Where
    the place? Upon the heath. There to meet with Macbeth. I come, Graymalkin.
    Paddock calls. Anon. All fair is foul, for the day has lost its name and night
    claims the hour. The raven himself is hoarse that croaks the fatal entrance of
    Duncan under my battlements. Come, you spirits that tend on mortal thoughts,
    unsex me here, and fill me from the crown to the toe top-full of direst cruelty.
    Make thick my blood, stop up the access and passage to remorse, that no
    compunctious visitings of nature shake my fell purpose, nor keep peace between
    the effect and it. Come to my woman's breasts and take my milk for gall, you
    murdering ministers, wherever in your sightless substances you wait on nature's
    mischief. Come thick night, and pall thee in the dunnest smoke of hell, that
    my keen knife see not the wound it makes, nor heaven peep through the blanket
    of the dark to cry Hold, Hold.
    """,
    """
    Now is the winter of our discontent made glorious summer by this sun of York,
    and all the clouds that loured upon our house in the deep bosom of the ocean
    buried. Now are our brows bound with victorious wreaths, our bruised arms hung
    up for monuments, our stern alarums changed to merry meetings, our dreadful
    marches to delightful measures. Grim-visaged war hath smoothed his wrinkled
    front, and now instead of mounting barbed steeds to fright the souls of fearful
    adversaries he capers nimbly in a lady's chamber to the lascivious pleasing
    of a lute. But I, that am not shaped for sportive tricks nor made to court an
    amorous looking-glass, I that am rudely stamped and want love's majesty to
    strut before a wanton ambling nymph, I that am curtailed of this fair proportion,
    cheated of feature by dissembling nature, deformed, unfinished, sent before my
    time into this breathing world scarce half made up, and that so lamely and
    unfashionable that dogs bark at me as I halt by them. Why I, in this weak
    piping time of peace, have no delight to pass away the time, unless to see my
    shadow in the sun and descant on mine own deformity. Therefore since I cannot
    prove a lover to entertain these fair well-spoken days, I am determined to prove
    a villain and hate the idle pleasures of these days.
    """,
    """
    The quality of mercy is not strained. It droppeth as the gentle rain from heaven
    upon the place beneath. It is twice blest, it blesseth him that gives and him
    that takes. It is mightiest in the mightiest, it becomes the throned monarch
    better than his crown. His sceptre shows the force of temporal power, the
    attribute to awe and majesty wherein doth sit the dread and fear of kings, but
    mercy is above this sceptred sway. It is enthroned in the hearts of kings, it
    is an attribute to God himself, and earthly power doth then show likest God's
    when mercy seasons justice. Therefore though justice be thy plea, consider this,
    that in the course of justice none of us should see salvation. We do pray for
    mercy, and that same prayer doth teach us all to render the deeds of mercy.
    I have spoke thus much to mitigate the justice of thy plea, which if thou
    follow, this strict court of Venice must needs give sentence against the merchant
    there. The pound of flesh which I demand of him is dearly bought, it is mine
    and I will have it. If you deny me, fie upon your law. There is no force in
    the decrees of Venice. I stand for judgment. Answer, shall I have it?
    """,
    """
    Out, out, brief candle. Life's but a walking shadow, a poor player that struts
    and frets his hour upon the stage and then is heard no more. It is a tale told
    by an idiot, full of sound and fury, signifying nothing. Tomorrow, and tomorrow,
    and tomorrow creeps in this petty pace from day to day to the last syllable of
    recorded time, and all our yesterdays have lighted fools the way to dusty death.
    Blow, wind, come wrack, at least we'll die with harness on our back. I have
    almost forgot the taste of fears. The time has been my senses would have cooled
    to hear a night-shriek, and my fell of hair would at a dismal treatise rouse
    and stir as life were in it. I have supped full with horrors. Direness, familiar
    to my slaughterous thoughts, cannot once start me. Wherefore was that cry? She
    should have died hereafter. There would have been a time for such a word. To
    morrow, and to morrow, and to morrow. What is done cannot be undone. We have
    scorched the snake not killed it. She'll close and be herself whilst our poor
    malice remains in danger of her former tooth. Things bad begun make strong
    themselves by ill.
    """,
    """
    Romeo, Romeo, wherefore art thou Romeo? Deny thy father and refuse thy name,
    or if thou wilt not, be but sworn my love and I'll no longer be a Capulet. It
    is only thy name that is my enemy. Thou art thyself, though not a Montague.
    What's Montague? It is nor hand nor foot nor arm nor face nor any other part
    belonging to a man. O be some other name. What's in a name? That which we call
    a rose by any other name would smell as sweet. So Romeo would, were he not
    Romeo called, retain that dear perfection which he owes without that title.
    Romeo, doff thy name, and for that name which is no part of thee take all
    myself. My bounty is as boundless as the sea, my love as deep. The more I give
    to thee the more I have, for both are infinite. Good night, good night. Parting
    is such sweet sorrow that I shall say good night till it be morrow. Love goes
    toward love as schoolboys from their books, but love from love toward school
    with heavy looks. Therefore love moderately, long love doth so, too swift
    arrives as tardy as too slow.
    """,
    """
    If music be the food of love, play on. Give me excess of it, that surfeiting,
    the appetite may sicken and so die. That strain again, it had a dying fall.
    O it came o'er my ear like the sweet sound that breathes upon a bank of violets,
    stealing and giving odour. Enough, no more, it is not so sweet now as it was
    before. O spirit of love how quick and fresh art thou, that notwithstanding
    thy capacity receiveth as the sea, nought enters there of what validity and
    pitch soever but falls into abatement and low price even with a minute. Love
    sought is good but given unsought is better. Some are born great, some achieve
    greatness, and some have greatness thrust upon them. Be not afraid of greatness.
    Some are born great, some achieve greatness, and others have greatness thrust
    upon them. Foolery sir does walk about the orb like the sun, it shines everywhere.
    I would give all my fame for a pot of ale and safety. Better a witty fool than
    a foolish wit. Many a good hanging prevents a bad marriage. There is no darkness
    but ignorance, in which life bettered the instruction.
    """,
    """
    We are such stuff as dreams are made on, and our little life is rounded with
    a sleep. Full fathom five thy father lies, of his bones are coral made, those
    are pearls that were his eyes, nothing of him that doth fade but doth suffer
    a sea-change into something rich and strange. Sea-nymphs hourly ring his knell.
    Hark now I hear them, ding-dong bell. Our revels now are ended. These our actors
    as I foretold you were all spirits and are melted into air, into thin air, and
    like the baseless fabric of this vision the cloud-capped towers, the gorgeous
    palaces, the solemn temples, the great globe itself, yea all which it inherit
    shall dissolve, and like this insubstantial pageant faded leave not a rack
    behind. We are such stuff as dreams are made on and our little life is rounded
    with a sleep. Sir, I am vexed. Bear with my weakness, my old brain is troubled.
    Be not disturbed with my infirmity. If you be pleased retire into my cell and
    there repose. A turn or two I'll walk to still my beating mind.
    """,
    """
    Once more unto the breach, dear friends, once more, or close the wall up with
    our English dead. In peace there's nothing so becomes a man as modest stillness
    and humility, but when the blast of war blows in our ears then imitate the
    action of the tiger. Stiffen the sinews, summon up the blood, disguise fair
    nature with hard-favoured rage. Then lend the eye a terrible aspect, let it
    pry through the portage of the head like the brass cannon, let the brow o'erwhelm
    it as fearfully as doth a galled rock o'erhang and jutty his confounded base,
    swilled with the wild and wasteful ocean. Now set the teeth and stretch the
    nostril wide, hold hard the breath and bend up every spirit to his full height.
    On, on, you noblest English whose blood is fet from fathers of war-proof, fathers
    that like so many Alexanders have in these parts from morn till even fought
    and sheathed their swords for lack of argument. Dishonour not your mothers, now
    attest that those whom you called fathers did beget you. Be copy now to men of
    grosser blood and teach them how to war. And you good yeomen whose limbs were
    made in England show us here the mettle of your pasture.
    """,
    """
    This royal throne of kings, this sceptred isle, this earth of majesty, this
    seat of Mars, this other Eden demi-paradise, this fortress built by Nature for
    herself against infection and the hand of war, this happy breed of men, this
    little world, this precious stone set in the silver sea which serves it in the
    office of a wall or as a moat defensive to a house against the envy of less
    happier lands, this blessed plot, this earth, this realm, this England. Dear
    for her reputation through the world is now leased out, like to a tenement or
    pelting farm. England bound in with the triumphant sea whose rocky shore beats
    back the envious siege of watery Neptune is now bound in with shame with
    inky blots and rotten parchment bonds. That England that was wont to conquer
    others hath made a shameful conquest of itself. Ah would the scandal vanish
    with my life how happy then were my ensuing death. For God's sake let us sit
    upon the ground and tell sad stories of the death of kings, how some have been
    deposed, some slain in war, some haunted by the ghosts they have deposed, some
    poisoned by their wives, some sleeping killed, all murdered.
    """,
]


def build_corpus(min_words: int = 10_000) -> str:
    """
    Concatenate passages until we reach at least min_words.
    Scene headers add variety so the model sees more than one repeated block.
    """
    parts: list[str] = []
    book = 1
    word_count = 0

    while word_count < min_words:
        parts.append(f"\n\n=== Book {book} ===\n\n")
        for scene, passage in enumerate(PASSAGES, start=1):
            parts.append(f"Scene {scene}\n")
            parts.append(passage.strip())
            parts.append("\n")
        book += 1
        word_count = len(" ".join(parts).split())

    return "".join(parts).strip()


if __name__ == "__main__":
    corpus = build_corpus()
    print(f"Corpus words: {len(corpus.split()):,}")
    print(f"Corpus chars: {len(corpus):,}")
