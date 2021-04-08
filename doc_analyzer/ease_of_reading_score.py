# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:19:47 2021

@author: ELECTROBOT
"""
# =============================================================================
# functions
# =============================================================================
import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|No)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|co|in)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text= re.sub(r'(?<=\[).+?(?=\])', "", text) # remove everything inside square brackets
    text= re.sub(r'(?<=\().+?(?=\))', "", text) # remove everything inside  brackets
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "i.e." in text: text = text.replace("i.e.","i<prd>e<prd>")
    if "e.g" in text: text = text.replace("e.g","e<prd>g")
    if "www." in text: text = text.replace("www.","www<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    #sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

easy_words=['a', 'able', 'aboard', 'about', 'above', 'absent', 'accept', 'accident', 'account', 'ache', 'aching', 'acorn', 'acre', 'across', 'act', 'acts', 'add', 'address', 'admire', 'adventure', 'afar', 'afraid', 'after', 'afternoon', 'afterward', 'afterwards', 'again', 'against', 'age', 'aged', 'ago', 'agree', 'ah', 'ahead', 'aid', 'aim', 'air', 'airfield', 'airplane', 'airport', 'airship', 'airy', 'alarm', 'alike', 'alive', 'all', 'alley', 'alligator', 'allow', 'almost', 'alone', 'along', 'aloud', 'already', 'also', 'always', 'am', 'america', 'american', 'among', 'amount', 'an', 'and', 'angel', 'anger', 'angry', 'animal', 'another', 'answer', 'ant', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'apart', 'apartment', 'ape', 'apiece', 'appear', 'apple', 'april', 'apron', 'are', "aren't", 'arise', 'arithmetic', 'arm', 'armful', 'army', 'arose', 'around', 'arrange', 'arrive', 'arrived', 'arrow', 'art', 'artist', 'as', 'ash', 'ashes', 'aside', 'ask', 'asleep', 'at', 'ate', 'attack', 'attend', 'attention', 'august', 'aunt', 'author', 'auto', 'automobile', 'autumn', 'avenue', 'awake', 'awaken', 'away', 'awful', 'awfully', 'awhile', 'ax', 'axe', 'baa', 'babe', 'babies', 'back', 'background', 'backward', 'backwards', 'bacon', 'bad', 'badge', 'badly', 'bag', 'bake', 'baker', 'bakery', 'baking', 'ball', 'balloon', 'banana', 'band', 'bandage', 'bang', 'banjo', 'bank', 'banker', 'bar', 'barber', 'bare', 'barefoot', 'barely', 'bark', 'barn', 'barrel', 'base', 'baseball', 'basement', 'basket', 'bat', 'batch', 'bath', 'bathe', 'bathing', 'bathroom', 'bathtub', 'battle', 'battleship', 'bay', 'be', 'beach', 'bead', 'beam', 'bean', 'bear', 'beard', 'beast', 'beat', 'beating', 'beautiful', 'beautify', 'beauty', 'became', 'because', 'become', 'becoming', 'bed', 'bedbug', 'bedroom', 'bedspread', 'bedtime', 'bee', 'beech', 'beef', 'beefsteak', 'beehive', 'been', 'beer', 'beet', 'before', 'beg', 'began', 'beggar', 'begged', 'begin', 'beginning', 'begun', 'behave', 'behind', 'being', 'believe', 'bell', 'belong', 'below', 'belt', 'bench', 'bend', 'beneath', 'bent', 'berries', 'berry', 'beside', 'besides', 'best', 'bet', 'better', 'between', 'bib', 'bible', 'bicycle', 'bid', 'big', 'bigger', 'bill', 'billboard', 'bin', 'bind', 'bird', 'birth', 'birthday', 'biscuit', 'bit', 'bite', 'biting', 'bitter', 'black', 'blackberry', 'blackbird', 'blackboard', 'blackness', 'blacksmith', 'blame', 'blank', 'blanket', 'blast', 'blaze', 'bleed', 'bless', 'blessing', 'blew', 'blind', 'blindfold', 'blinds', 'block', 'blood', 'bloom', 'blossom', 'blot', 'blow', 'blue', 'blueberry', 'bluebird', 'blush', 'board', 'boast', 'boat', 'bob', 'bobwhite', 'bodies', 'body', 'boil', 'boiler', 'bold', 'bone', 'bonnet', 'boo', 'book', 'bookcase', 'bookkeeper', 'boom', 'boot', 'born', 'borrow', 'boss', 'both', 'bother', 'bottle', 'bottom', 'bought', 'bounce', 'bow', 'bowl', 'bow-wow', 'box', 'boxcar', 'boxer', 'boxes', 'boy', 'boyhood', 'bracelet', 'brain', 'brake', 'bran', 'branch', 'brass', 'brave', 'bread', 'break', 'breakfast', 'breast', 'breath', 'breathe', 'breeze', 'brick', 'bride', 'bridge', 'bright', 'brightness', 'bring', 'broad', 'broadcast', 'broke', 'broken', 'brook', 'broom', 'brother', 'brought', 'brown', 'brush', 'bubble', 'bucket', 'buckle', 'bud', 'buffalo', 'bug', 'buggy', 'build', 'building', 'built', 'bulb', 'bull', 'bullet', 'bum', 'bumblebee', 'bump', 'bun', 'bunch', 'bundle', 'bunny', 'burn', 'burst', 'bury', 'bus', 'bush', 'bushel', 'business', 'busy', 'but', 'butcher', 'butt', 'butter', 'buttercup', 'butterfly', 'buttermilk', 'butterscotch', 'button', 'buttonhole', 'buy', 'buzz', 'by', 'bye', 'cab', 'cabbage', 'cabin', 'cabinet', 'cackle', 'cage', 'cake', 'calendar', 'calf', 'call', 'caller', 'calling', 'came', 'camel', 'camp', 'campfire', 'can', 'canal', 'canary', 'candle', 'candlestick', 'candy', 'cane', 'cannon', 'cannot', 'canoe', "can't", 'canyon', 'cap', 'cape', 'capital', 'captain', 'car', 'card', 'cardboard', 'care', 'careful', 'careless', 'carelessness', 'carload', 'carpenter', 'carpet', 'carriage', 'carrot', 'carry', 'cart', 'carve', 'case', 'cash', 'cashier', 'castle', 'cat', 'catbird', 'catch', 'catcher', 'caterpillar', 'catfish', 'catsup', 'cattle', 'caught', 'cause', 'cave', 'ceiling', 'cell', 'cellar', 'cent', 'center', 'cereal', 'certain', 'certainly', 'chain', 'chair', 'chalk', 'champion', 'chance', 'change', 'chap', 'charge', 'charm', 'chart', 'chase', 'chatter', 'cheap', 'cheat', 'check', 'checkers', 'cheek', 'cheer', 'cheese', 'cherry', 'chest', 'chew', 'chick', 'chicken', 'chief', 'child', 'childhood', 'children', 'chill', 'chilly', 'chimney', 'chin', 'china', 'chip', 'chipmunk', 'chocolate', 'choice', 'choose', 'chop', 'chorus', 'chose', 'chosen', 'christen', 'christmas', 'church', 'churn', 'cigarette', 'circle', 'circus', 'citizen', 'city', 'clang', 'clap', 'class', 'classmate', 'classroom', 'claw', 'clay', 'clean', 'cleaner', 'clear', 'clerk', 'clever', 'click', 'cliff', 'climb', 'clip', 'cloak', 'clock', 'close', 'closet', 'cloth', 'clothes', 'clothing', 'cloud', 'cloudy', 'clover', 'clown', 'club', 'cluck', 'clump', 'coach', 'coal', 'coast', 'coat', 'cob', 'cobbler', 'cocoa', 'coconut', 'cocoon', 'cod', 'codfish', 'coffee', 'coffeepot', 'coin', 'cold', 'collar', 'college', 'color', 'colored', 'colt', 'column', 'comb', 'come', 'comfort', 'comic', 'coming', 'company', 'compare', 'conductor', 'cone', 'connect', 'coo', 'cook', 'cooked', 'cooking', 'cookie', 'cookies', 'cool', 'cooler', 'coop', 'copper', 'copy', 'cord', 'cork', 'corn', 'corner', 'correct', 'cost', 'cot', 'cottage', 'cotton', 'couch', 'cough', 'could', "couldn't", 'count', 'counter', 'country', 'county', 'course', 'court', 'cousin', 'cover', 'cow', 'coward', 'cowardly', 'cowboy', 'cozy', 'crab', 'crack', 'cracker', 'cradle', 'cramps', 'cranberry', 'crank', 'cranky', 'crash', 'crawl', 'crazy', 'cream', 'creamy', 'creek', 'creep', 'crept', 'cried', 'croak', 'crook', 'crooked', 'crop', 'cross', 'crossing', 'cross-eyed', 'crow', 'crowd', 'crowded', 'crown', 'cruel', 'crumb', 'crumble', 'crush', 'crust', 'cry', 'cries', 'cub', 'cuff', 'cup', 'cuff', 'cup', 'cupboard', 'cupful', 'cure', 'curl', 'curly', 'curtain', 'curve', 'cushion', 'custard', 'customer', 'cut', 'cute', 'cutting', 'dab', 'dad', 'daddy', 'daily', 'dairy', 'daisy', 'dam', 'damage', 'dame', 'damp', 'dance', 'dancer', 'dancing', 'dandy', 'danger', 'dangerous', 'dare', 'dark', 'darkness', 'darling', 'darn', 'dart', 'dash', 'date', 'daughter', 'dawn', 'day', 'daybreak', 'daytime', 'dead', 'deaf', 'deal', 'dear', 'death', 'december', 'decide', 'deck', 'deed', 'deep', 'deer', 'defeat', 'defend', 'defense', 'delight', 'den', 'dentist', 'depend', 'deposit', 'describe', 'desert', 'deserve', 'desire', 'desk', 'destroy', 'devil', 'dew', 'diamond', 'did', "didn't", 'die', 'died', 'dies', 'difference', 'different', 'dig', 'dim', 'dime', 'dine', 'ding-dong', 'dinner', 'dip', 'direct', 'direction', 'dirt', 'dirty', 'discover', 'dish', 'dislike', 'dismiss', 'ditch', 'dive', 'diver', 'divide', 'do', 'dock', 'doctor', 'does', "doesn't", 'dog', 'doll', 'dollar', 'dolly', 'done', 'donkey', "don't", 'door', 'doorbell', 'doorknob', 'doorstep', 'dope', 'dot', 'double', 'dough', 'dove', 'down', 'downstairs', 'downtown', 'dozen', 'drag', 'drain', 'drank', 'draw', 'drawer', 'draw', 'drawing', 'dream', 'dress', 'dresser', 'dressmaker', 'drew', 'dried', 'drift', 'drill', 'drink', 'drip', 'drive', 'driven', 'driver', 'drop', 'drove', 'drown', 'drowsy', 'drub', 'drum', 'drunk', 'dry', 'duck', 'due', 'dug', 'dull', 'dumb', 'dump', 'during', 'dust', 'dusty', 'duty', 'dwarf', 'dwell', 'dwelt', 'dying', 'each', 'eager', 'eagle', 'ear', 'early', 'earn', 'earth', 'east', 'eastern', 'easy', 'eat', 'eaten', 'edge', 'egg', 'eh', 'eight', 'eighteen', 'eighth', 'eighty', 'either', 'elbow', 'elder', 'eldest', 'electric', 'electricity', 'elephant', 'eleven', 'elf', 'elm', 'else', 'elsewhere', 'empty', 'end', 'ending', 'enemy', 'engine', 'engineer', 'english', 'enjoy', 'enough', 'enter', 'envelope', 'equal', 'erase', 'eraser', 'errand', 'escape', 'eve', 'even', 'evening', 'ever', 'every', 'everybody', 'everyday', 'everyone', 'everything', 'everywhere', 'evil', 'exact', 'except', 'exchange', 'excited', 'exciting', 'excuse', 'exit', 'expect', 'explain', 'extra', 'eye', 'eyebrow', 'fable', 'face', 'facing', 'fact', 'factory', 'fail', 'faint', 'fair', 'fairy', 'faith', 'fake', 'fall', 'false', 'family', 'fan', 'fancy', 'far', 'faraway', 'fare', 'farmer', 'farm', 'farming', 'far-off', 'farther', 'fashion', 'fast', 'fasten', 'fat', 'father', 'fault', 'favor', 'favorite', 'fear', 'feast', 'feather', 'february', 'fed', 'feed', 'feel', 'feet', 'fell', 'fellow', 'felt', 'fence', 'fever', 'few', 'fib', 'fiddle', 'field', 'fife', 'fifteen', 'fifth', 'fifty', 'fig', 'fight', 'figure', 'file', 'fill', 'film', 'finally', 'find', 'fine', 'finger', 'finish', 'fire', 'firearm', 'firecracker', 'fireplace', 'fireworks', 'firing', 'first', 'fish', 'fisherman', 'fist', 'fit', 'fits', 'five', 'fix', 'flag', 'flake', 'flame', 'flap', 'flash', 'flashlight', 'flat', 'flea', 'flesh', 'flew', 'flies', 'flight', 'flip', 'flip-flop', 'float', 'flock', 'flood', 'floor', 'flop', 'flour', 'flow', 'flower', 'flowery', 'flutter', 'fly', 'foam', 'fog', 'foggy', 'fold', 'folks', 'follow', 'following', 'fond', 'food', 'fool', 'foolish', 'foot', 'football', 'footprint', 'for', 'forehead', 'forest', 'forget', 'forgive', 'forgot', 'forgotten', 'fork', 'form', 'fort', 'forth', 'fortune', 'forty', 'forward', 'fought', 'found', 'fountain', 'four', 'fourteen', 'fourth', 'fox', 'frame', 'free', 'freedom', 'freeze', 'freight', 'french', 'fresh', 'fret', 'friday', 'fried', 'friend', 'friendly', 'friendship', 'frighten', 'frog', 'from', 'front', 'frost', 'frown', 'froze', 'fruit', 'fry', 'fudge', 'fuel', 'full', 'fully', 'fun', 'funny', 'fur', 'furniture', 'further', 'fuzzy', 'gain', 'gallon', 'gallop', 'game', 'gang', 'garage', 'garbage', 'garden', 'gas', 'gasoline', 'gate', 'gather', 'gave', 'gay', 'gear', 'geese', 'general', 'gentle', 'gentleman', 'gentlemen', 'geography', 'get', 'getting', 'giant', 'gift', 'gingerbread', 'girl', 'give', 'given', 'giving', 'glad', 'gladly', 'glance', 'glass', 'glasses', 'gleam', 'glide', 'glory', 'glove', 'glow', 'glue', 'go', 'going', 'goes', 'goal', 'goat', 'gobble', 'god', 'god', 'godmother', 'gold', 'golden', 'goldfish', 'golf', 'gone', 'good', 'goods', 'goodbye', 'good-by', 'goodbye', 'good-bye', 'good-looking', 'goodness', 'goody', 'goose', 'gooseberry', 'got', 'govern', 'government', 'gown', 'grab', 'gracious', 'grade', 'grain', 'grand', 'grandchild', 'grandchildren', 'granddaughter', 'grandfather', 'grandma', 'grandmother', 'grandpa', 'grandson', 'grandstand', 'grape', 'grapes', 'grapefruit', 'grass', 'grasshopper', 'grateful', 'grave', 'gravel', 'graveyard', 'gravy', 'gray', 'graze', 'grease', 'great', 'green', 'greet', 'grew', 'grind', 'groan', 'grocery', 'ground', 'group', 'grove', 'grow', 'guard', 'guess', 'guest', 'guide', 'gulf', 'gum', 'gun', 'gunpowder', 'guy', 'ha', 'habit', 'had', "hadn't", 'hail', 'hair', 'haircut', 'hairpin', 'half', 'hall', 'halt', 'ham', 'hammer', 'hand', 'handful', 'handkerchief', 'handle', 'handwriting', 'hang', 'happen', 'happily', 'happiness', 'happy', 'harbor', 'hard', 'hardly', 'hardship', 'hardware', 'hare', 'hark', 'harm', 'harness', 'harp', 'harvest', 'has', "hasn't", 'haste', 'hasten', 'hasty', 'hat', 'hatch', 'hatchet', 'hate', 'haul', 'have', "haven't", 'having', 'hawk', 'hay', 'hayfield', 'haystack', 'he', 'head', 'headache', 'heal', 'health', 'healthy', 'heap', 'hear', 'hearing', 'heard', 'heart', 'heat', 'heater', 'heaven', 'heavy', "he'd", 'heel', 'height', 'held', 'hell', "he'll", 'hello', 'helmet', 'help', 'helper', 'helpful', 'hem', 'hen', 'henhouse', 'her', 'hers', 'herd', 'here', "here's", 'hero', 'herself', "he's", 'hey', 'hickory', 'hid', 'hidden', 'hide', 'high', 'highway', 'hill', 'hillside', 'hilltop', 'hilly', 'him', 'himself', 'hind', 'hint', 'hip', 'hire', 'his', 'hiss', 'history', 'hit', 'hitch', 'hive', 'ho', 'hoe', 'hog', 'hold', 'holder', 'hole', 'holiday', 'hollow', 'holy', 'home', 'homely', 'homesick', 'honest', 'honey', 'honeybee', 'honeymoon', 'honk', 'honor', 'hood', 'hoof', 'hook', 'hoop', 'hop', 'hope', 'hopeful', 'hopeless', 'horn', 'horse', 'horseback', 'horseshoe', 'hose', 'hospital', 'host', 'hot', 'hotel', 'hound', 'hour', 'house', 'housetop', 'housewife', 'housework', 'how', 'however', 'howl', 'hug', 'huge', 'hum', 'humble', 'hump', 'hundred', 'hung', 'hunger', 'hungry', 'hunk', 'hunt', 'hunter', 'hurrah', 'hurried', 'hurry', 'hurt', 'husband', 'hush', 'hut', 'hymn', 'i', 'ice', 'icy', "i'd", 'idea', 'ideal', 'if', 'ill', "i'll", "i'm", 'important', 'impossible', 'improve', 'in', 'inch', 'inches', 'income', 'indeed', 'indian', 'indoors', 'ink', 'inn', 'insect', 'inside', 'instant', 'instead', 'insult', 'intend', 'interested', 'interesting', 'into', 'invite', 'iron', 'is', 'island', "isn't", 'it', 'its', "it's", 'itself', "i've", 'ivory', 'ivy', 'jacket', 'jacks', 'jail', 'jam', 'january', 'jar', 'jaw', 'jay', 'jelly', 'jellyfish', 'jerk', 'jig', 'job', 'jockey', 'join', 'joke', 'joking', 'jolly', 'journey', 'joy', 'joyful', 'joyous', 'judge', 'jug', 'juice', 'juicy', 'july', 'jump', 'june', 'junior', 'junk', 'just', 'keen', 'keep', 'kept', 'kettle', 'key', 'kick', 'kid', 'kill', 'killed', 'kind', 'kindly', 'kindness', 'king', 'kingdom', 'kiss', 'kitchen', 'kite', 'kitten', 'kitty', 'knee', 'kneel', 'knew', 'knife', 'knit', 'knives', 'knob', 'knock', 'knot', 'know', 'known', 'lace', 'lad', 'ladder', 'ladies', 'lady', 'laid', 'lake', 'lamb', 'lame', 'lamp', 'land', 'lane', 'language', 'lantern', 'lap', 'lard', 'large', 'lash', 'lass', 'last', 'late', 'laugh', 'laundry', 'law', 'lawn', 'lawyer', 'lay', 'lazy', 'lead', 'leader', 'leaf', 'leak', 'lean', 'leap', 'learn', 'learned', 'least', 'leather', 'leave', 'leaving', 'led', 'left', 'leg', 'lemon', 'lemonade', 'lend', 'length', 'less', 'lesson', 'let', "let's", 'letter', 'letting', 'lettuce', 'level', 'liberty', 'library', 'lice', 'lick', 'lid', 'lie', 'life', 'lift', 'light', 'lightness', 'lightning', 'like', 'likely', 'liking', 'lily', 'limb', 'lime', 'limp', 'line', 'linen', 'lion', 'lip', 'list', 'listen', 'lit', 'little', 'live', 'lives', 'lively', 'liver', 'living', 'lizard', 'load', 'loaf', 'loan', 'loaves', 'lock', 'locomotive', 'log', 'lone', 'lonely', 'lonesome', 'long', 'look', 'lookout', 'loop', 'loose', 'lord', 'lose', 'loser', 'loss', 'lost', 'lot', 'loud', 'love', 'lovely', 'lover', 'low', 'luck', 'lucky', 'lumber', 'lump', 'lunch', 'lying', 'machine', 'machinery', 'mad', 'made', 'magazine', 'magic', 'maid', 'mail', 'mailbox', 'mailman', 'major', 'make', 'making', 'male', 'mama', 'mamma', 'man', 'manager', 'mane', 'manger', 'many', 'map', 'maple', 'marble', 'march', 'march', 'mare', 'mark', 'market', 'marriage', 'married', 'marry', 'mask', 'mast', 'master', 'mat', 'match', 'matter', 'mattress', 'may', 'may', 'maybe', 'mayor', 'maypole', 'me', 'meadow', 'meal', 'mean', 'means', 'meant', 'measure', 'meat', 'medicine', 'meet', 'meeting', 'melt', 'member', 'men', 'mend', 'meow', 'merry', 'mess', 'message', 'met', 'metal', 'mew', 'mice', 'middle', 'midnight', 'might', 'mighty', 'mile', 'milk', 'milkman', 'mill', 'miler', 'million', 'mind', 'mine', 'miner', 'mint', 'minute', 'mirror', 'mischief', 'miss', 'miss', 'misspell', 'mistake', 'misty', 'mitt', 'mitten', 'mix', 'moment', 'monday', 'money', 'monkey', 'month', 'moo', 'moon', 'moonlight', 'moose', 'mop', 'more', 'morning', 'morrow', 'moss', 'most', 'mostly', 'mother', 'motor', 'mount', 'mountain', 'mouse', 'mouth', 'move', 'movie', 'movies', 'moving', 'mow', 'mr.', 'mrs.', 'much', 'mud', 'muddy', 'mug', 'mule', 'multiply', 'murder', 'music', 'must', 'my', 'myself', 'nail', 'name', 'nap', 'napkin', 'narrow', 'nasty', 'naughty', 'navy', 'near', 'nearby', 'nearly', 'neat', 'neck', 'necktie', 'need', 'needle', "needn't", 'negro', 'neighbor', 'neighborhood', 'neither', 'nerve', 'nest', 'net', 'never', 'nevermore', 'new', 'news', 'newspaper', 'next', 'nibble', 'nice', 'nickel', 'night', 'nightgown', 'nine', 'nineteen', 'ninety', 'no', 'nobody', 'nod', 'noise', 'noisy', 'none', 'noon', 'nor', 'north', 'northern', 'nose', 'not', 'note', 'nothing', 'notice', 'november', 'now', 'nowhere', 'number', 'nurse', 'nut', 'oak', 'oar', 'oatmeal', 'oats', 'obey', 'ocean', "o'clock", 'october', 'odd', 'of', 'off', 'offer', 'office', 'officer', 'often', 'oh', 'oil', 'old', 'old-fashioned', 'on', 'once', 'one', 'onion', 'only', 'onward', 'open', 'or', 'orange', 'orchard', 'order', 'ore', 'organ', 'other', 'otherwise', 'ouch', 'ought', 'our', 'ours', 'ourselves', 'out', 'outdoors', 'outfit', 'outlaw', 'outline', 'outside', 'outward', 'oven', 'over', 'overalls', 'overcoat', 'overeat', 'overhead', 'overhear', 'overnight', 'overturn', 'owe', 'owing', 'owl', 'own', 'owner', 'ox', 'pa', 'pace', 'pack', 'package', 'pad', 'page', 'paid', 'pail', 'pain', 'painful', 'paint', 'painter', 'painting', 'pair', 'pal', 'palace', 'pale', 'pan', 'pancake', 'pane', 'pansy', 'pants', 'papa', 'paper', 'parade', 'pardon', 'parent', 'park', 'part', 'partly', 'partner', 'party', 'pass', 'passenger', 'past', 'paste', 'pasture', 'pat', 'patch', 'path', 'patter', 'pave', 'pavement', 'paw', 'pay', 'payment', 'pea', 'peas', 'peace', 'peaceful', 'peach', 'peaches', 'peak', 'peanut', 'pear', 'pearl', 'peck', 'peek', 'peel', 'peep', 'peg', 'pen', 'pencil', 'penny', 'people', 'pepper', 'peppermint', 'perfume', 'perhaps', 'person', 'pet', 'phone', 'piano', 'pick', 'pickle', 'picnic', 'picture', 'pie', 'piece', 'pig', 'pigeon', 'piggy', 'pile', 'pill', 'pillow', 'pin', 'pine', 'pineapple', 'pink', 'pint', 'pipe', 'pistol', 'pit', 'pitch', 'pitcher', 'pity', 'place', 'plain', 'plan', 'plane', 'plant', 'plate', 'platform', 'platter', 'play', 'player', 'playground', 'playhouse', 'playmate', 'plaything', 'pleasant', 'please', 'pleasure', 'plenty', 'plow', 'plug', 'plum', 'pocket', 'pocketbook', 'poem', 'point', 'poison', 'poke', 'pole', 'police', 'policeman', 'polish', 'polite', 'pond', 'ponies', 'pony', 'pool', 'poor', 'pop', 'popcorn', 'popped', 'porch', 'pork', 'possible', 'post', 'postage', 'postman', 'pot', 'potato', 'potatoes', 'pound', 'pour', 'powder', 'power', 'powerful', 'praise', 'pray', 'prayer', 'prepare', 'present', 'pretty', 'price', 'prick', 'prince', 'princess', 'print', 'prison', 'prize', 'promise', 'proper', 'protect', 'proud', 'prove', 'prune', 'public', 'puddle', 'puff', 'pull', 'pump', 'pumpkin', 'punch', 'punish', 'pup', 'pupil', 'puppy', 'pure', 'purple', 'purse', 'push', 'puss', 'pussy', 'pussycat', 'put', 'putting', 'puzzle', 'quack', 'quart', 'quarter', 'queen', 'queer', 'question', 'quick', 'quickly', 'quiet', 'quilt', 'quit', 'quite', 'rabbit', 'race', 'rack', 'radio', 'radish', 'rag', 'rail', 'railroad', 'railway', 'rain', 'rainy', 'rainbow', 'raise', 'raisin', 'rake', 'ram', 'ran', 'ranch', 'rang', 'rap', 'rapidly', 'rat', 'rate', 'rather', 'rattle', 'raw', 'ray', 'reach', 'read', 'reader', 'reading', 'ready', 'real', 'really', 'reap', 'rear', 'reason', 'rebuild', 'receive', 'recess', 'record', 'red', 'redbird', 'redbreast', 'refuse', 'reindeer', 'rejoice', 'remain', 'remember', 'remind', 'remove', 'rent', 'repair', 'repay', 'repeat', 'report', 'rest', 'return', 'review', 'reward', 'rib', 'ribbon', 'rice', 'rich', 'rid', 'riddle', 'ride', 'rider', 'riding', 'right', 'rim', 'ring', 'rip', 'ripe', 'rise', 'rising', 'river', 'road', 'roadside', 'roar', 'roast', 'rob', 'robber', 'robe', 'robin', 'rock', 'rocky', 'rocket', 'rode', 'roll', 'roller', 'roof', 'room', 'rooster', 'root', 'rope', 'rose', 'rosebud', 'rot', 'rotten', 'rough', 'round', 'route', 'row', 'rowboat', 'royal', 'rub', 'rubbed', 'rubber', 'rubbish', 'rug', 'rule', 'ruler', 'rumble', 'run', 'rung', 'runner', 'running', 'rush', 'rust', 'rusty', 'rye', 'sack', 'sad', 'saddle', 'sadness', 'safe', 'safety', 'said', 'sail', 'sailboat', 'sailor', 'saint', 'salad', 'sale', 'salt', 'same', 'sand', 'sandy', 'sandwich', 'sang', 'sank', 'sap', 'sash', 'sat', 'satin', 'satisfactory', 'saturday', 'sausage', 'savage', 'save', 'savings', 'saw', 'say', 'scab', 'scales', 'scare', 'scarf', 'school', 'schoolboy', 'schoolhouse', 'schoolmaster', 'schoolroom', 'scorch', 'score', 'scrap', 'scrape', 'scratch', 'scream', 'screen', 'screw', 'scrub', 'sea', 'seal', 'seam', 'search', 'season', 'seat', 'second', 'secret', 'see', 'seeing', 'seed', 'seek', 'seem', 'seen', 'seesaw', 'select', 'self', 'selfish', 'sell', 'send', 'sense', 'sent', 'sentence', 'separate', 'september', 'servant', 'serve', 'service', 'set', 'setting', 'settle', 'settlement', 'seven', 'seventeen', 'seventh', 'seventy', 'several', 'sew', 'shade', 'shadow', 'shady', 'shake', 'shaker', 'shaking', 'shall', 'shame', "shan't", 'shape', 'share', 'sharp', 'shave', 'she', "she'd", "she'll", "she's", 'shear', 'shears', 'shed', 'sheep', 'sheet', 'shelf', 'shell', 'shepherd', 'shine', 'shining', 'shiny', 'ship', 'shirt', 'shock', 'shoe', 'shoemaker', 'shone', 'shook', 'shoot', 'shop', 'shopping', 'shore', 'short', 'shot', 'should', 'shoulder', "shouldn't", 'shout', 'shovel', 'show', 'shower', 'shut', 'shy', 'sick', 'sickness', 'side', 'sidewalk', 'sideways', 'sigh', 'sight', 'sign', 'silence', 'silent', 'silk', 'sill', 'silly', 'silver', 'simple', 'sin', 'since', 'sing', 'singer', 'single', 'sink', 'sip', 'sir', 'sis', 'sissy', 'sister', 'sit', 'sitting', 'six', 'sixteen', 'sixth', 'sixty', 'size', 'skate', 'skater', 'ski', 'skin', 'skip', 'skirt', 'sky', 'slam', 'slap', 'slate', 'slave', 'sled', 'sleep', 'sleepy', 'sleeve', 'sleigh', 'slept', 'slice', 'slid', 'slide', 'sling', 'slip', 'slipped', 'slipper', 'slippery', 'slit', 'slow', 'slowly', 'sly', 'smack', 'small', 'smart', 'smell', 'smile', 'smoke', 'smooth', 'snail', 'snake', 'snap', 'snapping', 'sneeze', 'snow', 'snowy', 'snowball', 'snowflake', 'snuff', 'snug', 'so', 'soak', 'soap', 'sob', 'socks', 'sod', 'soda', 'sofa', 'soft', 'soil', 'sold', 'soldier', 'sole', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'son', 'song', 'soon', 'sore', 'sorrow', 'sorry', 'sort', 'soul', 'sound', 'soup', 'sour', 'south', 'southern', 'space', 'spade', 'spank', 'sparrow', 'speak', 'speaker', 'spear', 'speech', 'speed', 'spell', 'spelling', 'spend', 'spent', 'spider', 'spike', 'spill', 'spin', 'spinach', 'spirit', 'spit', 'splash', 'spoil', 'spoke', 'spook', 'spoon', 'sport', 'spot', 'spread', 'spring', 'springtime', 'sprinkle', 'square', 'squash', 'squeak', 'squeeze', 'squirrel', 'stable', 'stack', 'stage', 'stair', 'stall', 'stamp', 'stand', 'star', 'stare', 'start', 'starve', 'state', 'station', 'stay', 'steak', 'steal', 'steam', 'steamboat', 'steamer', 'steel', 'steep', 'steeple', 'steer', 'stem', 'step', 'stepping', 'stick', 'sticky', 'stiff', 'still', 'stillness', 'sting', 'stir', 'stitch', 'stock', 'stocking', 'stole', 'stone', 'stood', 'stool', 'stoop', 'stop', 'stopped', 'stopping', 'store', 'stork', 'stories', 'storm', 'stormy', 'story', 'stove', 'straight', 'strange', 'stranger', 'strap', 'straw', 'strawberry', 'stream', 'street', 'stretch', 'string', 'strip', 'stripes', 'strong', 'stuck', 'study', 'stuff', 'stump', 'stung', 'subject', 'such', 'suck', 'sudden', 'suffer', 'sugar', 'suit', 'sum', 'summer', 'sun', 'sunday', 'sunflower', 'sung', 'sunk', 'sunlight', 'sunny', 'sunrise', 'sunset', 'sunshine', 'supper', 'suppose', 'sure', 'surely', 'surface', 'surprise', 'swallow', 'swam', 'swamp', 'swan', 'swat', 'swear', 'sweat', 'sweater', 'sweep', 'sweet', 'sweetness', 'sweetheart', 'swell', 'swept', 'swift', 'swim', 'swimming', 'swing', 'switch', 'sword', 'swore', 'table', 'tablecloth', 'tablespoon', 'tablet', 'tack', 'tag', 'tail', 'tailor', 'take', 'taken', 'taking', 'tale', 'talk', 'talker', 'tall', 'tame', 'tan', 'tank', 'tap', 'tape', 'tar', 'tardy', 'task', 'taste', 'taught', 'tax', 'tea', 'teach', 'teacher', 'team', 'tear', 'tease', 'teaspoon', 'teeth', 'telephone', 'tell', 'temper', 'ten', 'tennis', 'tent', 'term', 'terrible', 'test', 'than', 'thank', 'thanks', 'thankful', 'thanksgiving', 'that', "that's", 'the', 'theater', 'thee', 'their', 'them', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'thick', 'thief', 'thimble', 'thin', 'thing', 'think', 'third', 'thirsty', 'thirteen', 'thirty', 'this', 'thorn', 'those', 'though', 'thought', 'thousand', 'thread', 'three', 'threw', 'throat', 'throne', 'through', 'throw', 'thrown', 'thumb', 'thunder', 'thursday', 'thy', 'tick', 'ticket', 'tickle', 'tie', 'tiger', 'tight', 'till', 'time', 'tin', 'tinkle', 'tiny', 'tip', 'tiptoe', 'tire', 'tired', 'title', 'to', 'toad', 'toadstool', 'toast', 'tobacco', 'today', 'toe', 'together', 'toilet', 'told', 'tomato', 'tomorrow', 'ton', 'tone', 'tongue', 'tonight', 'too', 'took', 'tool', 'toot', 'tooth', 'toothbrush', 'toothpick', 'top', 'tore', 'torn', 'toss', 'touch', 'tow', 'toward', 'towards', 'towel', 'tower', 'town', 'toy', 'trace', 'track', 'trade', 'train', 'tramp', 'trap', 'tray', 'treasure', 'treat', 'tree', 'trick', 'tricycle', 'tried', 'trim', 'trip', 'trolley', 'trouble', 'truck', 'true', 'truly', 'trunk', 'trust', 'truth', 'try', 'tub', 'tuesday', 'tug', 'tulip', 'tumble', 'tune', 'tunnel', 'turkey', 'turn', 'turtle', 'twelve', 'twenty', 'twice', 'twig', 'twin', 'two', 'ugly', 'umbrella', 'uncle', 'under', 'understand', 'underwear', 'undress', 'unfair', 'unfinished', 'unfold', 'unfriendly', 'unhappy', 'unhurt', 'uniform', 'united', 'states', 'unkind', 'unknown', 'unless', 'unpleasant', 'until', 'unwilling', 'up', 'upon', 'upper', 'upset', 'upside', 'upstairs', 'uptown', 'upward', 'us', 'use', 'used', 'useful', 'valentine', 'valley', 'valuable', 'value', 'vase', 'vegetable', 'velvet', 'very', 'vessel', 'victory', 'view', 'village', 'vine', 'violet', 'visit', 'visitor', 'voice', 'vote', 'wag', 'wagon', 'waist', 'wait', 'wake', 'waken', 'walk', 'wall', 'walnut', 'want', 'war', 'warm', 'warn', 'was', 'wash', 'washer', 'washtub', "wasn't", 'waste', 'watch', 'watchman', 'water', 'watermelon', 'waterproof', 'wave', 'wax', 'way', 'wayside', 'we', 'weak', 'weakness', 'weaken', 'wealth', 'weapon', 'wear', 'weary', 'weather', 'weave', 'web', "we'd", 'wedding', 'wednesday', 'wee', 'weed', 'week', "we'll", 'weep', 'weigh', 'welcome', 'well', 'went', 'were', "we're", 'west', 'western', 'wet', "we've", 'whale', 'what', "what's", 'wheat', 'wheel', 'when', 'whenever', 'where', 'which', 'while', 'whip', 'whipped', 'whirl', 'whisky', 'whiskey', 'whisper', 'whistle', 'white', 'who', "who'd", 'whole', "who'll", 'whom', "who's", 'whose', 'why', 'wicked', 'wide', 'wife', 'wiggle', 'wild', 'wildcat', 'will', 'willing', 'willow', 'win', 'wind', 'windy', 'windmill', 'window', 'wine', 'wing', 'wink', 'winner', 'winter', 'wipe', 'wire', 'wise', 'wish', 'wit', 'witch', 'with', 'without', 'woke', 'wolf', 'woman', 'women', 'won', 'wonder', 'wonderful', "won't", 'wood', 'wooden', 'woodpecker', 'woods', 'wool', 'woolen', 'word', 'wore', 'work', 'worker', 'workman', 'world', 'worm', 'worn', 'worry', 'worse', 'worst', 'worth', 'would', "wouldn't", 'wound', 'wove', 'wrap', 'wrapped', 'wreck', 'wren', 'wring', 'write', 'writing', 'written', 'wrong', 'wrote', 'wrung', 'yard', 'yarn', 'year', 'yell', 'yellow', 'yes', 'yesterday', 'yet', 'yolk', 'yonder', 'you', "you'd", "you'll", 'young', 'youngster', 'your', 'yours', "you're", 'yourself', 'yourselves', 'youth', "you've"]

# import spacy
# nlp= spacy.load(r"C:/Users/ELECTROBOT/Anaconda3/envs/bot/Lib/site-packages/en_core_web_sm/en_core_web_sm-2.3.1")
# from textstat.textstat import textstatistics, easy_word_set, legacy_round
  
# Splits the text into sentences, using 
# Spacy's sentence segmentation which can 
# be found at https://spacy.io/usage/spacy-101
  
# Returns Number of Words in the text
def word_count(text):
    sentences = split_into_sentences(text)
    words = 0
    for sentence in sentences:
        words += len([token for token in sentence.split()])
    return words
  
# Returns the number of sentences in the text
def sentence_count(text):
    sentences = split_into_sentences(text)
    return len(sentences)
  
# Returns average sentence length
def avg_sentence_length(text):
    words = word_count(text)
    sentences = sentence_count(text)
    average_sentence_length = float(words / sentences)
    return average_sentence_length
  
# Textstat is a python package, to calculate statistics from 
# text to determine readability, 
# complexity and grade level of a particular corpus.
# Package can be found at https://pypi.python.org/pypi/textstat
def syllables_count(text):
    sentences = split_into_sentences(text)
    syllables = 0
    for sentence in sentences:
        syllables += len([token for token in sentence])
    return syllables
  
# Returns the average number of syllables per
# word in the text
def avg_syllables_per_word(text):
    syllable = syllables_count(text)
    words = word_count(text)
    ASPW = float(syllable) / float(words)
    return round(ASPW, 1)
  
# Return total Difficult Words in a text
def difficult_words(text):
  
    # Find all words in the text
    words = []
    sentences = split_into_sentences(text)
    for sentence in sentences:
        words += [str(token) for token in sentence.split()]
  
    # difficult words are those with syllables >= 2
    # easy_word_set is provide by Textstat as 
    # a list of common words
    diff_words_set = set()
      
    for word in words:
        syllable_count = syllables_count(word)
        if word not in easy_words and syllable_count >= 2:
            diff_words_set.add(word)
  
    return len(diff_words_set)
  
# A word is polysyllablic if it has more than 3 syllables
# this functions returns the number of all such words 
# present in the text
def poly_syllable_count(text):
    count = 0
    words = []
    sentences = split_into_sentences(text)
    for sentence in sentences:
        words += [token for token in sentence.split()]
      
  
    for word in words:
        syllable_count = syllables_count(word)
        if syllable_count >= 3:
            count += 1
    return count
  

# =============================================================================
# FleschKincaidTest
# =============================================================================
def FleschKincaidTest(text):
	score = 0.0
	if len(text) > 0:
		score = (0.39 * len(text.split()) / len(text.split('.')) ) + 11.8 * ( sum(list(map(lambda x: 1 if x in ["a","i","e","o","u","y","A","E","I","O","U","y"] else 0,text))) / len(text.split())) - 15.59
		return score if score > 0 else 0
    

#90-100--easy
#0-30--defficult

# =============================================================================
# ari
# =============================================================================
def ari(text):
    
    words=word_count(text)
    sentences=sentence_count(text)
    characters=syllables_count(text)
    score=4.71*(characters/words) + .5*(words/sentences) -21.43
    return score
# 1	5-6	Kindergarten
# 2	6-7	First/Second Grade
# 3	7-9	Third Grade
# 4	9-10	Fourth Grade
# 5	10-11	Fifth Grade
# 6	11-12	Sixth Grade
# 7	12-13	Seventh Grade
# 8	13-14	Eighth Grade
# 9	14-15	Ninth Grade
# 10	15-16	Tenth Grade
# 11	16-17	Eleventh Grade
# 12	17-18	Twelfth grade
# 13	18-24	College student
# 14	24+	Professor


# =============================================================================
# dale 
#formula:  .1579*(difficult_words/words)*100 + .0496*(words/sentences)
# =============================================================================
def dale_chall_readability_score(text):
    """
        Implements Dale Challe Formula:
        Raw score = 0.1579*(PDW) + 0.0496*(ASL) + 3.6365
        Here,
            PDW = Percentage of difficult words.
            ASL = Average sentence length
    """
    words = word_count(text)
    # Number of words not termed as difficult words
    count = words - difficult_words(text)
    if words > 0:
  
        # Percentage of words not on difficult word list
  
        per = float(count) / float(words) * 100
      
    # diff_words stores percentage of difficult words
    diff_words = 100 - per
  
    raw_score = (0.1579 * diff_words) + \
                (0.0496 * avg_sentence_length(text))
      
    # If Percentage of Difficult Words is greater than 5 %, then;
    # Adjusted Score = Raw Score + 3.6365,
    # otherwise Adjusted Score = Raw Score
  
    if diff_words > 5:       
  
        raw_score += 3.6365
          
    return round(raw_score, 2)



# 4.9 or lower	easily understood by an average 4th-grade student or lower
# 5.0–5.9	easily understood by an average 5th or 6th-grade student
# 6.0–6.9	easily understood by an average 7th or 8th-grade student
# 7.0–7.9	easily understood by an average 9th or 10th-grade student
# 8.0–8.9	easily understood by an average 11th or 12th-grade student
# 9.0–9.9	easily understood by an average 13th to 15th-grade (college) student