# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:28:11 2021

@author: ELECTROBOT
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
import spacy
nlp= spacy.load('en_core_web_sm')
os.chdir(r'C:\Users\ELECTROBOT\Desktop\kaggle\commonlit_readbility')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train= pd.read_csv('train_june13.csv')

# =============================================================================
# #create reading parameters
# =============================================================================

#sent length by number of sents

def sent_len(text):
    text2=re.sub(r'[^\w\s]','', text)
    doc= nlp(text2)
    sents= doc.sents
    all_len=[]
    for sent in sents:
        words= len(sent.text.split())
        all_len.append(words)
    return sum(all_len)/len(all_len)



#number of easy dale words

dale_chall_words="a|able|aboard|about|above|absent|accept|accident|account|ache|aching|acorn|acre|across|act|acts|add|address|admire|adventure|afar|afraid|after|afternoon|afterward|afterwards|again|against|age|aged|ago|agree|ah|ahead|aid|aim|air|airfield|airplane|airport|airship|airy|alarm|alike|alive|all|alley|alligator|allow|almost|alone|along|aloud|already|also|always|am|America|American|among|amount|an|and|angel|anger|angry|animal|another|answer|ant|any|anybody|anyhow|anyone|anything|anyway|anywhere|apart|apartment|ape|apiece|appear|apple|April|apron|are|aren't|arise|arithmetic|arm|armful|army|arose|around|arrange|arrive|arrived|arrow|art|artist|as|ash|ashes|aside|ask|asleep|at|ate|attack|attend|attention|August|aunt|author|auto|automobile|autumn|avenue|awake|awaken|away|awful|awfully|awhile|ax|axe|baa|babe|babies|back|background|backward|backwards|bacon|bad|badge|badly|bag|bake|baker|bakery|baking|ball|balloon|banana|band|bandage|bang|banjo|bank|banker|bar|barber|bare|barefoot|barely|bark|barn|barrel|base|baseball|basement|basket|bat|batch|bath|bathe|bathing|bathroom|bathtub|battle|battleship|bay|be|beach|bead|beam|bean|bear|beard|beast|beat|beating|beautiful|beautify|beauty|became|because|become|becoming|bed|bedbug|bedroom|bedspread|bedtime|bee|beech|beef|beefsteak|beehive|been|beer|beet|before|beg|began|beggar|begged|begin|beginning|begun|behave|behind|being|believe|bell|belong|below|belt|bench|bend|beneath|bent|berries|berry|beside|besides|best|bet|better|between|bib|bible|bicycle|bid|big|bigger|bill|billboard|bin|bind|bird|birth|birthday|biscuit|bit|bite|biting|bitter|black|blackberry|blackbird|blackboard|blackness|blacksmith|blame|blank|blanket|blast|blaze|bleed|bless|blessing|blew|blind|blindfold|blinds|block|blood|bloom|blossom|blot|blow|blue|blueberry|bluebird|blush|board|boast|boat|bob|bobwhite|bodies|body|boil|boiler|bold|bone|bonnet|boo|book|bookcase|bookkeeper|boom|boot|born|borrow|boss|both|bother|bottle|bottom|bought|bounce|bow|bowl|bow-wow|box|boxcar|boxer|boxes|boy|boyhood|bracelet|brain|brake|bran|branch|brass|brave|bread|break|breakfast|breast|breath|breathe|breeze|brick|bride|bridge|bright|brightness|bring|broad|broadcast|broke|broken|brook|broom|brother|brought|brown|brush|bubble|bucket|buckle|bud|buffalo|bug|buggy|build|building|built|bulb|bull|bullet|bum|bumblebee|bump|bun|bunch|bundle|bunny|burn|burst|bury|bus|bush|bushel|business|busy|but|butcher|butt|butter|buttercup|butterfly|buttermilk|butterscotch|button|buttonhole|buy|buzz|by|bye|cab|cabbage|cabin|cabinet|cackle|cage|cake|calendar|calf|call|caller|calling|came|camel|camp|campfire|can|canal|canary|candle|candlestick|candy|cane|cannon|cannot|canoe|can't|canyon|cap|cape|capital|captain|car|card|cardboard|care|careful|careless|carelessness|carload|carpenter|carpet|carriage|carrot|carry|cart|carve|case|cash|cashier|castle|cat|catbird|catch|catcher|caterpillar|catfish|catsup|cattle|caught|cause|cave|ceiling|cell|cellar|cent|center|cereal|certain|certainly|chain|chair|chalk|champion|chance|change|chap|charge|charm|chart|chase|chatter|cheap|cheat|check|checkers|cheek|cheer|cheese|cherry|chest|chew|chick|chicken|chief|child|childhood|children|chill|chilly|chimney|chin|china|chip|chipmunk|chocolate|choice|choose|chop|chorus|chose|chosen|christen|Christmas|church|churn|cigarette|circle|circus|citizen|city|clang|clap|class|classmate|classroom|claw|clay|clean|cleaner|clear|clerk|clever|click|cliff|climb|clip|cloak|clock|close|closet|cloth|clothes|clothing|cloud|cloudy|clover|clown|club|cluck|clump|coach|coal|coast|coat|cob|cobbler|cocoa|coconut|cocoon|cod|codfish|coffee|coffeepot|coin|cold|collar|college|color|colored|colt|column|comb|come|comfort|comic|coming|company|compare|conductor|cone|connect|coo|cook|cooked|cooking|cookie|cookies|cool|cooler|coop|copper|copy|cord|cork|corn|corner|correct|cost|cot|cottage|cotton|couch|cough|could|couldn't|count|counter|country|county|course|court|cousin|cover|cow|coward|cowardly|cowboy|cozy|crab|crack|cracker|cradle|cramps|cranberry|crank|cranky|crash|crawl|crazy|cream|creamy|creek|creep|crept|cried|croak|crook|crooked|crop|cross|crossing|cross-eyed|crow|crowd|crowded|crown|cruel|crumb|crumble|crush|crust|cry|cries|cub|cuff|cup|cuff|cup|cupboard|cupful|cure|curl|curly|curtain|curve|cushion|custard|customer|cut|cute|cutting|dab|dad|daddy|daily|dairy|daisy|dam|damage|dame|damp|dance|dancer|dancing|dandy|danger|dangerous|dare|dark|darkness|darling|darn|dart|dash|date|daughter|dawn|day|daybreak|daytime|dead|deaf|deal|dear|death|December|decide|deck|deed|deep|deer|defeat|defend|defense|delight|den|dentist|depend|deposit|describe|desert|deserve|desire|desk|destroy|devil|dew|diamond|did|didn't|die|died|dies|difference|different|dig|dim|dime|dine|ding-dong|dinner|dip|direct|direction|dirt|dirty|discover|dish|dislike|dismiss|ditch|dive|diver|divide|do|dock|doctor|does|doesn't|dog|doll|dollar|dolly|done|donkey|don't|door|doorbell|doorknob|doorstep|dope|dot|double|dough|dove|down|downstairs|downtown|dozen|drag|drain|drank|draw|drawer|draw|drawing|dream|dress|dresser|dressmaker|drew|dried|drift|drill|drink|drip|drive|driven|driver|drop|drove|drown|drowsy|drub|drum|drunk|dry|duck|due|dug|dull|dumb|dump|during|dust|dusty|duty|dwarf|dwell|dwelt|dying|each|eager|eagle|ear|early|earn|earth|east|eastern|easy|eat|eaten|edge|egg|eh|eight|eighteen|eighth|eighty|either|elbow|elder|eldest|electric|electricity|elephant|eleven|elf|elm|else|elsewhere|empty|end|ending|enemy|engine|engineer|English|enjoy|enough|enter|envelope|equal|erase|eraser|errand|escape|eve|even|evening|ever|every|everybody|everyday|everyone|everything|everywhere|evil|exact|except|exchange|excited|exciting|excuse|exit|expect|explain|extra|eye|eyebrow|fable|face|facing|fact|factory|fail|faint|fair|fairy|faith|fake|fall|false|family|fan|fancy|far|faraway|fare|farmer|farm|farming|far-off|farther|fashion|fast|fasten|fat|father|fault|favor|favorite|fear|feast|feather|February|fed|feed|feel|feet|fell|fellow|felt|fence|fever|few|fib|fiddle|field|fife|fifteen|fifth|fifty|fig|fight|figure|file|fill|film|finally|find|fine|finger|finish|fire|firearm|firecracker|fireplace|fireworks|firing|first|fish|fisherman|fist|fit|fits|five|fix|flag|flake|flame|flap|flash|flashlight|flat|flea|flesh|flew|flies|flight|flip|flip-flop|float|flock|flood|floor|flop|flour|flow|flower|flowery|flutter|fly|foam|fog|foggy|fold|folks|follow|following|fond|food|fool|foolish|foot|football|footprint|for|forehead|forest|forget|forgive|forgot|forgotten|fork|form|fort|forth|fortune|forty|forward|fought|found|fountain|four|fourteen|fourth|fox|frame|free|freedom|freeze|freight|French|fresh|fret|Friday|fried|friend|friendly|friendship|frighten|frog|from|front|frost|frown|froze|fruit|fry|fudge|fuel|full|fully|fun|funny|fur|furniture|further|fuzzy|gain|gallon|gallop|game|gang|garage|garbage|garden|gas|gasoline|gate|gather|gave|gay|gear|geese|general|gentle|gentleman|gentlemen|geography|get|getting|giant|gift|gingerbread|girl|give|given|giving|glad|gladly|glance|glass|glasses|gleam|glide|glory|glove|glow|glue|go|going|goes|goal|goat|gobble|God|god|godmother|gold|golden|goldfish|golf|gone|good|goods|goodbye|good-by|goodbye|good-bye|good-looking|goodness|goody|goose|gooseberry|got|govern|government|gown|grab|gracious|grade|grain|grand|grandchild|grandchildren|granddaughter|grandfather|grandma|grandmother|grandpa|grandson|grandstand|grape|grapes|grapefruit|grass|grasshopper|grateful|grave|gravel|graveyard|gravy|gray|graze|grease|great|green|greet|grew|grind|groan|grocery|ground|group|grove|grow|guard|guess|guest|guide|gulf|gum|gun|gunpowder|guy|ha|habit|had|hadn't|hail|hair|haircut|hairpin|half|hall|halt|ham|hammer|hand|handful|handkerchief|handle|handwriting|hang|happen|happily|happiness|happy|harbor|hard|hardly|hardship|hardware|hare|hark|harm|harness|harp|harvest|has|hasn't|haste|hasten|hasty|hat|hatch|hatchet|hate|haul|have|haven't|having|hawk|hay|hayfield|haystack|he|head|headache|heal|health|healthy|heap|hear|hearing|heard|heart|heat|heater|heaven|heavy|he'd|heel|height|held|hell|he'll|hello|helmet|help|helper|helpful|hem|hen|henhouse|her|hers|herd|here|here's|hero|herself|he's|hey|hickory|hid|hidden|hide|high|highway|hill|hillside|hilltop|hilly|him|himself|hind|hint|hip|hire|his|hiss|history|hit|hitch|hive|ho|hoe|hog|hold|holder|hole|holiday|hollow|holy|home|homely|homesick|honest|honey|honeybee|honeymoon|honk|honor|hood|hoof|hook|hoop|hop|hope|hopeful|hopeless|horn|horse|horseback|horseshoe|hose|hospital|host|hot|hotel|hound|hour|house|housetop|housewife|housework|how|however|howl|hug|huge|hum|humble|hump|hundred|hung|hunger|hungry|hunk|hunt|hunter|hurrah|hurried|hurry|hurt|husband|hush|hut|hymn|I|ice|icy|I'd|idea|ideal|if|ill|I'll|I'm|important|impossible|improve|in|inch|inches|income|indeed|Indian|indoors|ink|inn|insect|inside|instant|instead|insult|intend|interested|interesting|into|invite|iron|is|island|isn't|it|its|it's|itself|I've|ivory|ivy|jacket|jacks|jail|jam|January|jar|jaw|jay|jelly|jellyfish|jerk|jig|job|jockey|join|joke|joking|jolly|journey|joy|joyful|joyous|judge|jug|juice|juicy|July|jump|June|junior|junk|just|keen|keep|kept|kettle|key|kick|kid|kill|killed|kind|kindly|kindness|king|kingdom|kiss|kitchen|kite|kitten|kitty|knee|kneel|knew|knife|knit|knives|knob|knock|knot|know|known|lace|lad|ladder|ladies|lady|laid|lake|lamb|lame|lamp|land|lane|language|lantern|lap|lard|large|lash|lass|last|late|laugh|laundry|law|lawn|lawyer|lay|lazy|lead|leader|leaf|leak|lean|leap|learn|learned|least|leather|leave|leaving|led|left|leg|lemon|lemonade|lend|length|less|lesson|let|let's|letter|letting|lettuce|level|liberty|library|lice|lick|lid|lie|life|lift|light|lightness|lightning|like|likely|liking|lily|limb|lime|limp|line|linen|lion|lip|list|listen|lit|little|live|lives|lively|liver|living|lizard|load|loaf|loan|loaves|lock|locomotive|log|lone|lonely|lonesome|long|look|lookout|loop|loose|lord|lose|loser|loss|lost|lot|loud|love|lovely|lover|low|luck|lucky|lumber|lump|lunch|lying|ma|machine|machinery|mad|made|magazine|magic|maid|mail|mailbox|mailman|major|make|making|male|mama|mamma|man|manager|mane|manger|many|map|maple|marble|march|March|mare|mark|market|marriage|married|marry|mask|mast|master|mat|match|matter|mattress|may|May|maybe|mayor|maypole|me|meadow|meal|mean|means|meant|measure|meat|medicine|meet|meeting|melt|member|men|mend|meow|merry|mess|message|met|metal|mew|mice|middle|midnight|might|mighty|mile|milk|milkman|mill|miler|million|mind|mine|miner|mint|minute|mirror|mischief|miss|Miss|misspell|mistake|misty|mitt|mitten|mix|moment|Monday|money|monkey|month|moo|moon|moonlight|moose|mop|more|morning|morrow|moss|most|mostly|mother|motor|mount|mountain|mouse|mouth|move|movie|movies|moving|mow|Mr.|Mrs.|much|mud|muddy|mug|mule|multiply|murder|music|must|my|myself|nail|name|nap|napkin|narrow|nasty|naughty|navy|near|nearby|nearly|neat|neck|necktie|need|needle|needn't|Negro|neighbor|neighborhood|neither|nerve|nest|net|never|nevermore|new|news|newspaper|next|nibble|nice|nickel|night|nightgown|nine|nineteen|ninety|no|nobody|nod|noise|noisy|none|noon|nor|north|northern|nose|not|note|nothing|notice|November|now|nowhere|number|nurse|nut|oak|oar|oatmeal|oats|obey|ocean|o'clock|October|odd|of|off|offer|office|officer|often|oh|oil|old|old-fashioned|on|once|one|onion|only|onward|open|or|orange|orchard|order|ore|organ|other|otherwise|ouch|ought|our|ours|ourselves|out|outdoors|outfit|outlaw|outline|outside|outward|oven|over|overalls|overcoat|overeat|overhead|overhear|overnight|overturn|owe|owing|owl|own|owner|ox|pa|pace|pack|package|pad|page|paid|pail|pain|painful|paint|painter|painting|pair|pal|palace|pale|pan|pancake|pane|pansy|pants|papa|paper|parade|pardon|parent|park|part|partly|partner|party|pass|passenger|past|paste|pasture|pat|patch|path|patter|pave|pavement|paw|pay|payment|pea|peas|peace|peaceful|peach|peaches|peak|peanut|pear|pearl|peck|peek|peel|peep|peg|pen|pencil|penny|people|pepper|peppermint|perfume|perhaps|person|pet|phone|piano|pick|pickle|picnic|picture|pie|piece|pig|pigeon|piggy|pile|pill|pillow|pin|pine|pineapple|pink|pint|pipe|pistol|pit|pitch|pitcher|pity|place|plain|plan|plane|plant|plate|platform|platter|play|player|playground|playhouse|playmate|plaything|pleasant|please|pleasure|plenty|plow|plug|plum|pocket|pocketbook|poem|point|poison|poke|pole|police|policeman|polish|polite|pond|ponies|pony|pool|poor|pop|popcorn|popped|porch|pork|possible|post|postage|postman|pot|potato|potatoes|pound|pour|powder|power|powerful|praise|pray|prayer|prepare|present|pretty|price|prick|prince|princess|print|prison|prize|promise|proper|protect|proud|prove|prune|public|puddle|puff|pull|pump|pumpkin|punch|punish|pup|pupil|puppy|pure|purple|purse|push|puss|pussy|pussycat|put|putting|puzzle|quack|quart|quarter|queen|queer|question|quick|quickly|quiet|quilt|quit|quite|rabbit|race|rack|radio|radish|rag|rail|railroad|railway|rain|rainy|rainbow|raise|raisin|rake|ram|ran|ranch|rang|rap|rapidly|rat|rate|rather|rattle|raw|ray|reach|read|reader|reading|ready|real|really|reap|rear|reason|rebuild|receive|recess|record|red|redbird|redbreast|refuse|reindeer|rejoice|remain|remember|remind|remove|rent|repair|repay|repeat|report|rest|return|review|reward|rib|ribbon|rice|rich|rid|riddle|ride|rider|riding|right|rim|ring|rip|ripe|rise|rising|river|road|roadside|roar|roast|rob|robber|robe|robin|rock|rocky|rocket|rode|roll|roller|roof|room|rooster|root|rope|rose|rosebud|rot|rotten|rough|round|route|row|rowboat|royal|rub|rubbed|rubber|rubbish|rug|rule|ruler|rumble|run|rung|runner|running|rush|rust|rusty|rye|sack|sad|saddle|sadness|safe|safety|said|sail|sailboat|sailor|saint|salad|sale|salt|same|sand|sandy|sandwich|sang|sank|sap|sash|sat|satin|satisfactory|Saturday|sausage|savage|save|savings|saw|say|scab|scales|scare|scarf|school|schoolboy|schoolhouse|schoolmaster|schoolroom|scorch|score|scrap|scrape|scratch|scream|screen|screw|scrub|sea|seal|seam|search|season|seat|second|secret|see|seeing|seed|seek|seem|seen|seesaw|select|self|selfish|sell|send|sense|sent|sentence|separate|September|servant|serve|service|set|setting|settle|settlement|seven|seventeen|seventh|seventy|several|sew|shade|shadow|shady|shake|shaker|shaking|shall|shame|shan't|shape|share|sharp|shave|she|she'd|she'll|she's|shear|shears|shed|sheep|sheet|shelf|shell|shepherd|shine|shining|shiny|ship|shirt|shock|shoe|shoemaker|shone|shook|shoot|shop|shopping|shore|short|shot|should|shoulder|shouldn't|shout|shovel|show|shower|shut|shy|sick|sickness|side|sidewalk|sideways|sigh|sight|sign|silence|silent|silk|sill|silly|silver|simple|sin|since|sing|singer|single|sink|sip|sir|sis|sissy|sister|sit|sitting|six|sixteen|sixth|sixty|size|skate|skater|ski|skin|skip|skirt|sky|slam|slap|slate|slave|sled|sleep|sleepy|sleeve|sleigh|slept|slice|slid|slide|sling|slip|slipped|slipper|slippery|slit|slow|slowly|sly|smack|small|smart|smell|smile|smoke|smooth|snail|snake|snap|snapping|sneeze|snow|snowy|snowball|snowflake|snuff|snug|so|soak|soap|sob|socks|sod|soda|sofa|soft|soil|sold|soldier|sole|some|somebody|somehow|someone|something|sometime|sometimes|somewhere|son|song|soon|sore|sorrow|sorry|sort|soul|sound|soup|sour|south|southern|space|spade|spank|sparrow|speak|speaker|spear|speech|speed|spell|spelling|spend|spent|spider|spike|spill|spin|spinach|spirit|spit|splash|spoil|spoke|spook|spoon|sport|spot|spread|spring|springtime|sprinkle|square|squash|squeak|squeeze|squirrel|stable|stack|stage|stair|stall|stamp|stand|star|stare|start|starve|state|station|stay|steak|steal|steam|steamboat|steamer|steel|steep|steeple|steer|stem|step|stepping|stick|sticky|stiff|still|stillness|sting|stir|stitch|stock|stocking|stole|stone|stood|stool|stoop|stop|stopped|stopping|store|stork|stories|storm|stormy|story|stove|straight|strange|stranger|strap|straw|strawberry|stream|street|stretch|string|strip|stripes|strong|stuck|study|stuff|stump|stung|subject|such|suck|sudden|suffer|sugar|suit|sum|summer|sun|Sunday|sunflower|sung|sunk|sunlight|sunny|sunrise|sunset|sunshine|supper|suppose|sure|surely|surface|surprise|swallow|swam|swamp|swan|swat|swear|sweat|sweater|sweep|sweet|sweetness|sweetheart|swell|swept|swift|swim|swimming|swing|switch|sword|swore|table|tablecloth|tablespoon|tablet|tack|tag|tail|tailor|take|taken|taking|tale|talk|talker|tall|tame|tan|tank|tap|tape|tar|tardy|task|taste|taught|tax|tea|teach|teacher|team|tear|tease|teaspoon|teeth|telephone|tell|temper|ten|tennis|tent|term|terrible|test|than|thank|thanks|thankful|Thanksgiving|that|that's|the|theater|thee|their|them|then|there|these|they|they'd|they'll|they're|they've|thick|thief|thimble|thin|thing|think|third|thirsty|thirteen|thirty|this|thorn|those|though|thought|thousand|thread|three|threw|throat|throne|through|throw|thrown|thumb|thunder|Thursday|thy|tick|ticket|tickle|tie|tiger|tight|till|time|tin|tinkle|tiny|tip|tiptoe|tire|tired|title|to|toad|toadstool|toast|tobacco|today|toe|together|toilet|told|tomato|tomorrow|ton|tone|tongue|tonight|too|took|tool|toot|tooth|toothbrush|toothpick|top|tore|torn|toss|touch|tow|toward|towards|towel|tower|town|toy|trace|track|trade|train|tramp|trap|tray|treasure|treat|tree|trick|tricycle|tried|trim|trip|trolley|trouble|truck|true|truly|trunk|trust|truth|try|tub|Tuesday|tug|tulip|tumble|tune|tunnel|turkey|turn|turtle|twelve|twenty|twice|twig|twin|two|ugly|umbrella|uncle|under|understand|underwear|undress|unfair|unfinished|unfold|unfriendly|unhappy|unhurt|uniform|United States|unkind|unknown|unless|unpleasant|until|unwilling|up|upon|upper|upset|upside|upstairs|uptown|upward|us|use|used|useful|valentine|valley|valuable|value|vase|vegetable|velvet|very|vessel|victory|view|village|vine|violet|visit|visitor|voice|vote|wag|wagon|waist|wait|wake|waken|walk|wall|walnut|want|war|warm|warn|was|wash|washer|washtub|wasn't|waste|watch|watchman|water|watermelon|waterproof|wave|wax|way|wayside|we|weak|weakness|weaken|wealth|weapon|wear|weary|weather|weave|web|we'd|wedding|Wednesday|wee|weed|week|we'll|weep|weigh|welcome|well|went|were|we're|west|western|wet|we've|whale|what|what's|wheat|wheel|when|whenever|where|which|while|whip|whipped|whirl|whisky|whiskey|whisper|whistle|white|who|who'd|whole|who'll|whom|who's|whose|why|wicked|wide|wife|wiggle|wild|wildcat|will|willing|willow|win|wind|windy|windmill|window|wine|wing|wink|winner|winter|wipe|wire|wise|wish|wit|witch|with|without|woke|wolf|woman|women|won|wonder|wonderful|won't|wood|wooden|woodpecker|woods|wool|woolen|word|wore|work|worker|workman|world|worm|worn|worry|worse|worst|worth|would|wouldn't|wound|wove|wrap|wrapped|wreck|wren|wring|write|writing|written|wrong|wrote|wrung|yard|yarn|year|yell|yellow|yes|yesterday|yet|yolk|yonder|you|you'd|you'll|young|youngster|your|yours|you're|yourself|yourselves|youth|you've"
dale_chall_list=dale_chall_words.split('|')


#number of hardwords according to phenomes
from nltk.corpus import cmudict
d = cmudict.dict()

def nsyl(word):
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]
    except KeyError:
        #if word not found in cmudict
        return syllables(word)

def syllables(word):
    #referred from stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word
    count = 0
    vowels = 'aeiouy'
    word = word.lower()
    if word[0] in vowels:
        count +=1
    for index in range(1,len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count +=1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count += 1
    if count == 0:
        count += 1
    return count

def hard_word(text):
    doc= nlp(text)
    sents= doc.sents
    total_score=[]
    for sent in sents:
        #print(sent)
    # Initiating hard and total words 
        hard_words=0
        count=0
        
        text2=re.sub(r'[^\w\s]','', sent.text) 
        for i in text2.split(' ') :
            try:
                count+=1
                ns=nsyl(i.lower())
                if type(ns)==int :
                    if ns>2 :
                        hard_words+=1
                else:
                    if ns[0]>2 :
                        hard_words+=1
            except:
                pass
        score= hard_words/count
        total_score.append(score)
    return sum(total_score)/len(total_score)



#scores based on the occurance of 'CONJ', 'CCONJ', 'VERB' in the sentence. Also extended sentences with PRON. 

def spacy_trial(text):
    try:
        sentences= text.split('.')
        all_scores=[]
        for num,sent in enumerate(sentences):
            
            sent=nlp(sent)
            if len(sent)>0:
                definers=[i for i in sent if i.pos_ in ['CONJ', 'CCONJ', 'VERB']]
                definers2=[i for num,i in enumerate(sent) if i.pos_=="PRON" and sent[num].is_title==False]
                
                [definers.append(i) for i in definers2]
                
                score= len(definers)/len(sent.text.split())
                all_scores.append(score)
                
        return sum(all_scores)/len(all_scores)
    except:
        return 0
        
    
# =============================================================================
# Apply on train
# =============================================================================

train['avg_sent_len']= 0
train['dale_words']= 0
train['hard_word']= 0
train['spacy_trial']= 0

def apply_reading(text):
    avg_sent_len= sent_len(text)
    dale_word= len([i for i in text.split() if i not in dale_chall_list])
    hardword= hard_word(text)
    spacytrial= spacy_trial(text)
    
    return avg_sent_len,dale_word,hardword,spacytrial

for i in range(0, len(train)):
    print(".", end="", flush=True)
    train['avg_sent_len'].iloc[i],train['dale_words'].iloc[i],train['hard_word'].iloc[i],train['spacy_trial'].iloc[i]= apply_reading(train[ 'excerpt'].iloc[i])
    

# =============================================================================
# #scale the new attributes
# =============================================================================

# dat= train[['avg_sent_len','dale_words','hard_word','spacy_trial']]


# from sklearn.preprocessing import StandardScaler
# std= StandardScaler()
# std.fit(dat)


# df= pd.DataFrame(std.transform(dat), columns= list(dat))

#cor=df.corr()
train['avg_sent_len']=[(i-min(train['avg_sent_len']))/(max(train['avg_sent_len'])-min(train['avg_sent_len'])) for i in train['avg_sent_len']]
train['dale_words']=[(i-min(train['dale_words']))/(max(train['dale_words'])-min(train['dale_words'])) for i in train['dale_words']]
train['hard_word']=[(i-min(train['hard_word']))/(max(train['hard_word'])-min(train['hard_word'])) for i in train['hard_word']]
train['spacy_trial']=[(i-min(train['spacy_trial']))/(max(train['spacy_trial'])-min(train['spacy_trial'])) for i in train['spacy_trial']]




#induce standard error in the target

#train['final_target']= [i+i*j for i,j in zip(train.target, train.standard_error)]


#train.to_csv('train_july6.csv', index=False)
# import matplotlib.pyplot as plt
# plt.hist(train['target']+1)



# =============================================================================
# create model for prediction
# =============================================================================

from transformers import AdamW
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup

path=r"C:\Users\ELECTROBOT\Desktop\roberta_base"
tokenizer = AutoTokenizer.from_pretrained(path)

#df=train.copy()

# =============================================================================
# data loader
# =============================================================================

MAX_LEN=256
#data loader
class loader(torch.utils.data.Dataset):
    def __init__(self, df, test=True):
        super().__init__()

        self.df = df        
        self.test = test
        self.text = df.excerpt.tolist()
        self.avg_sent_len= df.avg_sent_len.tolist()
        self.dale_words= df.dale_words.tolist()
        self.hard_word= df.hard_word.tolist()
        self.spacy_trial= df.spacy_trial.tolist()
        #self.text = [text.replace("\n", " ") for text in self.text]
        
        if not self.test:
            self.target = torch.tensor(df.target.values, dtype=torch.float32)
    
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )        
 

    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        attention_mask = torch.tensor(self.encoded['attention_mask'][index])
        avg_sent_len=torch.tensor(self.avg_sent_len[index])
        dale_words=torch.tensor(self.dale_words[index])
        hard_word=torch.tensor(self.hard_word[index])
        spacy_trial=torch.tensor(self.spacy_trial[index])
        
        if self.test:
            return (input_ids, attention_mask,avg_sent_len,dale_words,hard_word,spacy_trial)            
        else:
            target = self.target[index]
            return (input_ids, attention_mask, avg_sent_len,dale_words,hard_word,spacy_trial,target)




#dataset=loader(train, test=False)
#train_loader=torch.utils.data.DataLoader(dataset, batch_size=4,drop_last=True, shuffle=True)


# for batch in train_loader:
#     break
# input_ids, attention_mask, avg_sent_len,dale_words,hard_word,spacy_trial,target=batch

# =============================================================================
# model architecture
# =============================================================================

class Model_rob(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(path)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       
        
        self.roberta = AutoModel.from_pretrained(path, config=config)  
            
        self.attention = nn.Sequential(            
            nn.Linear(768, 512),            
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )        
        self.fc1= nn.Sequential(nn.Linear(768,1))
        self.relu =  nn.ReLU()
        self.tanh =  nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.bn11 = nn.BatchNorm1d(14)
        self.regressor = nn.Sequential(                        
            nn.Linear(5, 1)                        
        )
        

    def forward(self, input_ids, attention_mask,avg_sent_len,dale_words,hard_word,spacy_trial):
        roberta_output = self.roberta(input_ids=input_ids,attention_mask=attention_mask)
        
        last_layer_hidden_states = roberta_output.hidden_states[-1]

        weights = self.attention(last_layer_hidden_states)
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)        
        
        
        red_contect_vector= self.fc1(context_vector)
        
        x=torch.cat((red_contect_vector,avg_sent_len.view(avg_sent_len.shape[0],1)),1)
        
        x=torch.cat((x,dale_words.view(dale_words.shape[0],1)),1)
        
        x=torch.cat((x,hard_word.view(hard_word.shape[0],1)),1)
        
        x=torch.cat((x,spacy_trial.view(spacy_trial.shape[0],1)),1)
        #x= self.bn11(x)
        
        x= self.relu(x)
        
        # Now we reduce the context vector to the prediction score.
        return self.regressor(x)

# =============================================================================
# model architecture without reading scores
# =============================================================================
# class Model_rob(nn.Module):
#     def __init__(self):
#         super().__init__()

#         config = AutoConfig.from_pretrained(path)
#         config.update({"output_hidden_states":True, 
#                         "hidden_dropout_prob": 0.0,
#                         "layer_norm_eps": 1e-7})                       
        
#         self.roberta = AutoModel.from_pretrained(path, config=config)  
            
#         self.attention = nn.Sequential(            
#             nn.Linear(768, 512),            
#             nn.Tanh(),                       
#             nn.Linear(512, 1),
#             nn.Softmax(dim=1)
#         )        

#         self.regressor = nn.Sequential(                        
#             nn.Linear(768, 1)                        
#         )
        

#     def forward(self, input_ids, attention_mask,avg_sent_len,dale_words,hard_word,spacy_trial):
#         roberta_output = self.roberta(input_ids=input_ids,attention_mask=attention_mask)        
#         last_layer_hidden_states = roberta_output.hidden_states[-1]
#         weights = self.attention(last_layer_hidden_states)
#         context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)        
#         return self.regressor(context_vector)


# =============================================================================
# 
# =============================================================================

# #loss
# def eval_mse(model, data_loader):
#     """Evaluates the mean squared error of the |model| on |data_loader|"""
#     model.eval()            
#     mse_sum = 0

#     with torch.no_grad():
#         for batch_num, (input_ids, attention_mask, target) in enumerate(data_loader):
#             input_ids = input_ids.to(DEVICE)
#             attention_mask = attention_mask.to(DEVICE)                        
#             target = target.to(DEVICE)           
            
#             pred = model(input_ids, attention_mask)                       

#             mse_sum += nn.MSELoss(reduction="sum")(pred.flatten(), target).item()
                

#     return mse_sum / len(data_loader.dataset)


# #predict
# def predict(model, data_loader):
#     """Returns an np.array with predictions of the |model| on |data_loader|"""
#     model.eval()

#     result = np.zeros(len(data_loader.dataset))    
#     index = 0
    
#     with torch.no_grad():
#         for batch_num, (input_ids, attention_mask) in enumerate(data_loader):
#             input_ids = input_ids.to(DEVICE)
#             attention_mask = attention_mask.to(DEVICE)
                        
#             pred = model(input_ids, attention_mask)                        

#             result[index : index + pred.shape[0]] = pred.flatten().to("cpu")
#             index += pred.shape[0]

#     return result





# #train
# def train(model, model_path, train_loader, val_loader,
#           optimizer, scheduler=None, num_epochs=NUM_EPOCHS):    
    
#     best_epoch = 0
#     step = 0
#     total_loss=0
#     total_preds=[]


#     for epoch in range(num_epochs):                           
#         val_rmse = None         

#         for batch_num, (input_ids, attention_mask, avg_sent_len,dale_words,hard_word,spacy_trial,target) in enumerate(train_loader):
            
#             input_ids = input_ids.to(DEVICE)
#             attention_mask = attention_mask.to(DEVICE)      
#             avg_sent_len = avg_sent_len.to(DEVICE)      
#             dale_words = dale_words.to(DEVICE)      
#             hard_word = hard_word.to(DEVICE)      
#             spacy_trial = spacy_trial.to(DEVICE)      
#             target = target.to(DEVICE)      

#             optimizer.zero_grad()
            
#             model.train()
            
#             pred = model(input_ids, attention_mask)
                                                        
#             #mseloss = nn.MSELoss()
#             #loss=mseloss(pred.view(-1), target)
    
            
#             mse = nn.MSELoss(reduction="mean")(pred.flatten(), target)
#             total_loss = total_loss + mse.item()
#             mse.backward()

#             optimizer.step()
#             if scheduler:
#                 scheduler.step()
            
#             pred=pred.detach().cpu().numpy()
    
#             # append the model predictions
#             total_preds.append(pred)
        
#         avg_loss = total_loss / len(train_loader)
#         total_preds  = np.concatenate(total_preds, axis=0)




# =============================================================================
# train function
# =============================================================================

def train_model():
  
  model.train()

  total_loss, total_accuracy = 0, 0
  
  # empty list to save model predictions
  total_preds=[]
  
  # iterate over batches
  for step, batch in enumerate(train_loader):
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_loader)))
      
    # push the batch to gpu
    batch = [r.to(device) for r in batch]
    
    input_ids, attention_mask, avg_sent_len,dale_words,hard_word,spacy_trial,target = batch
    
    # clear previously calculated gradients 
    model.zero_grad()
    
    # get model predictions for the current batch
    preds = model(input_ids, attention_mask, avg_sent_len,dale_words,hard_word,spacy_trial)
    
    # compute the loss between actual and predicted values
    mseloss = nn.MSELoss()
    loss=mseloss(preds.view(-1), target)
    
    
    # add on to the total loss
    total_loss = total_loss + loss.item()
    
    # backward pass to calculate the gradients
    loss.backward()
    
    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # update parameters
    optimizer.step()
    
    # model predictions are stored on GPU. So, push it to CPU
    preds=preds.detach().cpu().numpy()
    
    # append the model predictions
    total_preds.append(preds)
    
  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_loader)
  
  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  #returns the loss and predictions
  return avg_loss, total_preds


# =============================================================================
# evaluate function
# =============================================================================
def evaluate():
  
  print("\nEvaluating...")
  
  # deactivate dropout layers
  model.eval()

  total_loss = 0
  
  # empty list to save the model predictions
  total_preds = []

  # iterate over batches
  for step, batch in enumerate(val_loader):
      #print(step)
      #break
    
    # Progress update every 50 batches.
    if step % 50 == 0 and not step == 0:
      
      # Calculate elapsed time in minutes.
      #elapsed = format_time(time.time() - t0)
            
      # Report progress.
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_loader)))

    # push the batch to gpu
    batch = [t.to(device) for t in batch]

    input_ids, attention_mask, avg_sent_len,dale_words,hard_word,spacy_trial,target = batch

    # deactivate autograd
    with torch.no_grad():
      
      # model predictions
      preds = model(input_ids, attention_mask, avg_sent_len,dale_words,hard_word,spacy_trial)
      mseloss = nn.MSELoss()
      loss=mseloss(preds.view(-1), target)
      # compute the validation loss between actual and predicted values
      #loss = cross_entropy(preds,labels)

      total_loss = total_loss + loss.item()

      preds = preds.detach().cpu().numpy()

      total_preds.append(preds)

  # compute the validation loss of the epoch
  avg_loss = total_loss / len(val_loader) 

  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  return avg_loss, total_preds





# =============================================================================
# fiune tune model
# =============================================================================

# =============================================================================
# call model and optimizer
# =============================================================================

# model = Model_rob()
# model = model.to(device)
from transformers import AdamW
import time
from sklearn.model_selection import KFold
# # define the optimizer
# optimizer = AdamW(model.parameters(),lr = .0001)    

#optimizer = torch.optim.SGD(model.parameters(), lr =.0001 )




# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]
epochs=10
models_saved=[]


kfold = KFold(n_splits=5, random_state=10, shuffle=True)

start = time.time()
for fold, (train_indices, val_indices) in enumerate(kfold.split(train)): 
    
    print(f"\nFold {fold + 1}/{5}")
    #break
    
    train_df= train.loc[train_indices]
    dataset=loader(train_df, test=False)
    train_loader=torch.utils.data.DataLoader(dataset, batch_size=4,drop_last=True, shuffle=True)

    val_df= train.loc[val_indices]
    dataset=loader(val_df, test=False)
    val_loader=torch.utils.data.DataLoader(dataset, batch_size=4,drop_last=True, shuffle=False)
    
    model = Model_rob().to(device)
    optimizer = AdamW(model.parameters(),lr = .0001)  
    
    
    for epoch in range(epochs):
     
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
        
        train_loss, _ = train_model()
    
        
        valid_loss, _ = evaluate()
        
        #save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            #torch.save(model.state_dict(), 'saved_weights_full_model.pt')
            torch.save(model.state_dict(), 'final_model/model_saved_tanh.pt')
            models_saved.append((fold,epoch))
        
        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')
        
end = time.time()   

(end-start)/60 #42 mins


# =============================================================================
# load best model
# =============================================================================



#load weights of best model
path = 'final_model/model_saved_tanh.pt'
model.load_state_dict(torch.load(path))

# =============================================================================
# predict
# =============================================================================

#get valdiatoin df from fold where best model was saved#

dataset=loader(val_df, test=False)
test_loader=torch.utils.data.DataLoader(dataset, batch_size=1,drop_last=True, shuffle=False)



def predict():
  
  print("\npredicting...")
  
  # deactivate dropout layers
  
  total_preds=[]# iterate over batches
  for step,batch in enumerate(test_loader):
    print(".", end="", flush=True)
    # push the batch to gpu
    batch = [t.to(device) for t in batch]
    
    input_ids, attention_mask, avg_sent_len,dale_words,hard_word,spacy_trial,target = batch
    
    # deactivate autograd
    with torch.no_grad():
        
        # model predictions
        preds = model(input_ids, attention_mask, avg_sent_len,dale_words,hard_word,spacy_trial)
        
        preds = preds.detach().cpu().numpy()
        
        [total_preds.append(i[0]) for i in preds]
  return total_preds


preds= predict()


tt=pd.DataFrame({'text': val_df.excerpt, 'actual':val_df['target'], 'preds': preds})
val_df['pred']=preds







            


#cerate optimizer


def create_optimizer(model):
    named_parameters = list(model.named_parameters())    
    
    roberta_parameters = named_parameters[:197]    
    attention_parameters = named_parameters[199:203]
    regressor_parameters = named_parameters[203:]
        
    attention_group = [params for (name, params) in attention_parameters]
    regressor_group = [params for (name, params) in regressor_parameters]

    parameters = []
    parameters.append({"params": attention_group})
    parameters.append({"params": regressor_group})

    for layer_num, (name, params) in enumerate(roberta_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01

        lr = 2e-5

        if layer_num >= 69:        
            lr = 5e-5

        if layer_num >= 133:
            lr = 1e-4

        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})

    return AdamW(parameters)
