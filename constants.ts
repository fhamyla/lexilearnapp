
import { Difficulty, TracingCategory, TracingItem, ReadingItem, SpellingItem, MemoryItem } from './types';

// ==========================================
// 40 UNIQUE TRACING LEVELS
// All paths converted to Cubic Bezier (C) or Lines (L) 
// to ensure accurate hit detection in the tracing engine.
// ==========================================

const createAdaptive = (id: string, cat: TracingCategory, label: string, path: string) => {
  return {
    id,
    category: cat,
    label: label,
    pathData: path,
    viewBox: '0 0 300 300',
    difficultyConfig: {
      [Difficulty.PROFOUND]: { label: label, pathData: path },
      [Difficulty.SEVERE]: { label: label, pathData: path },
      [Difficulty.MODERATE]: { label: label, pathData: path },
      [Difficulty.MILD]: { label: label, pathData: path },
    }
  };
};

// Helper for Circle via Bezier (approximate)
// M 150,50 ...
const CIRCLE_BEZIER = 'M 150,50 C 205,50 250,95 250,150 C 250,205 205,250 150,250 C 95,250 50,205 50,150 C 50,95 95,50 150,50';

const ALPHABET = [
  { l: 'A', d: 'M 50,250 L 150,50 L 250,250 M 100,150 L 200,150' },
  { l: 'B', d: 'M 80,50 L 80,250 M 80,50 C 180,50 180,150 80,150 C 180,150 180,250 80,250' },
  // Smooth C: Wide opening for easier tracing
  { l: 'C', d: 'M 220,70 C 180,30 100,30 80,100 C 60,180 100,260 180,260 C 220,260 230,220 230,220' }, 
  { l: 'D', d: 'M 80,50 L 80,250 M 80,50 C 220,50 220,250 80,250' },
  { l: 'E', d: 'M 220,50 L 80,50 L 80,250 L 220,250 M 80,150 L 200,150' },
  { l: 'F', d: 'M 220,50 L 80,50 L 80,250 M 80,150 L 180,150' },
  { l: 'G', d: 'M 220,60 C 180,30 100,30 80,100 C 60,180 100,240 150,240 C 200,240 220,200 220,150 L 150,150' }, 
  { l: 'H', d: 'M 80,50 L 80,250 M 220,50 L 220,250 M 80,150 L 220,150' },
  { l: 'I', d: 'M 150,50 L 150,250 M 100,50 L 200,50 M 100,250 L 200,250' },
  { l: 'J', d: 'M 200,50 L 200,200 C 200,240 150,260 100,240' },
  { l: 'K', d: 'M 80,50 L 80,250 M 220,50 L 80,150 L 220,250' },
  { l: 'L', d: 'M 80,50 L 80,250 L 220,250' },
  { l: 'M', d: 'M 50,250 L 50,50 L 150,150 L 250,50 L 250,250' },
  { l: 'N', d: 'M 80,250 L 80,50 L 220,250 L 220,50' },
  { l: 'O', d: CIRCLE_BEZIER },
  { l: 'P', d: 'M 80,250 L 80,50 M 80,50 C 200,50 200,150 80,150' },
  { l: 'Q', d: CIRCLE_BEZIER + ' M 180,180 L 250,250' },
  { l: 'R', d: 'M 80,250 L 80,50 M 80,50 C 200,50 200,150 80,150 M 140,150 L 220,250' },
  { l: 'S', d: 'M 220,60 C 180,40 80,40 80,100 C 80,160 220,140 220,200 C 220,260 120,260 80,240' },
  { l: 'T', d: 'M 150,50 L 150,250 M 50,50 L 250,50' },
  { l: 'U', d: 'M 80,50 L 80,200 C 80,240 220,240 220,200 L 220,50' },
  { l: 'V', d: 'M 50,50 L 150,250 L 250,50' },
  { l: 'W', d: 'M 50,50 L 80,250 L 150,100 L 220,250 L 250,50' },
  { l: 'X', d: 'M 50,50 L 250,250 M 250,50 L 50,250' },
  { l: 'Y', d: 'M 50,50 L 150,150 L 250,50 M 150,150 L 150,250' },
  { l: 'Z', d: 'M 50,50 L 250,50 L 50,250 L 250,250' }
];

const NUMBERS = [
  { l: '0', d: 'M 150,50 C 205,50 250,95 250,150 C 250,205 205,250 150,250 C 95,250 50,205 50,150 C 50,95 95,50 150,50' }, // Oval
  { l: '1', d: 'M 120,80 L 150,50 L 150,250' },
  { l: '2', d: 'M 60,100 C 60,40 240,40 240,120 C 240,180 60,250 60,250 L 250,250' },
  { l: '3', d: 'M 60,60 L 240,60 L 160,140 C 240,140 250,240 150,250 C 100,250 60,220 60,220' }, 
  { l: '4', d: 'M 200,250 L 200,50 L 50,200 L 250,200' },
  { l: '5', d: 'M 230,60 L 90,60 L 80,140 C 80,140 100,120 160,120 C 230,120 230,240 150,240 C 100,240 70,220 70,220' }, 
  { l: '6', d: 'M 200,50 C 100,50 60,150 60,150 C 60,220 100,250 150,250 C 200,250 220,200 220,180 C 220,140 180,140 150,140 C 100,140 80,160 80,160' },
  { l: '7', d: 'M 60,60 L 240,60 L 100,250' },
  { l: '8', d: 'M 150,150 C 100,150 80,120 80,90 C 80,40 220,40 220,90 C 220,120 200,150 150,150 C 200,150 220,180 220,210 C 220,260 80,260 80,210 C 80,180 100,150 150,150' },
  // FIXED NUMBER 9: Standard "ball and stick" (Top Circle -> Line Down)
  { l: '9', d: 'M 200,100 C 200,127 177,150 150,150 C 122,150 100,127 100,100 C 100,72 122,50 150,50 C 177,50 200,72 200,100 L 200,250' },
];

export const TRACING_ITEMS: TracingItem[] = [
  // 1-4: BASICS
  createAdaptive('t1', TracingCategory.LINES, 'Vertical Line', 'M 150,50 L 150,250'),
  createAdaptive('t2', TracingCategory.LINES, 'Horizontal Line', 'M 50,150 L 250,150'),
  createAdaptive('t3', TracingCategory.SHAPES, 'Circle', CIRCLE_BEZIER),
  createAdaptive('t4', TracingCategory.LINES, 'Cross', 'M 150,50 L 150,250 M 50,150 L 250,150'),

  // 5-30: LETTERS A-Z
  ...ALPHABET.map((item, i) => createAdaptive(`t${i+5}`, TracingCategory.LETTERS, `Letter ${item.l}`, item.d)),

  // 31-40: NUMBERS 0-9
  ...NUMBERS.map((item, i) => createAdaptive(`t${i+31}`, TracingCategory.NUMBERS, `Number ${item.l}`, item.d)),
];


// ==========================================
// READING LEVELS
// ==========================================

const createAdaptiveWord = (id: string, level: number, variants: { [key in Difficulty]: { w: string, s: string } }) => ({
  id,
  difficultyLevel: level,
  word: variants[Difficulty.MILD].w, // Default
  sentence: variants[Difficulty.MILD].s,
  difficultyConfig: {
    [Difficulty.PROFOUND]: { word: variants[Difficulty.PROFOUND].w, sentence: variants[Difficulty.PROFOUND].s },
    [Difficulty.SEVERE]: { word: variants[Difficulty.SEVERE].w, sentence: variants[Difficulty.SEVERE].s },
    [Difficulty.MODERATE]: { word: variants[Difficulty.MODERATE].w, sentence: variants[Difficulty.MODERATE].s },
    [Difficulty.MILD]: { word: variants[Difficulty.MILD].w, sentence: variants[Difficulty.MILD].s },
  }
});

export const READING_ITEMS: ReadingItem[] = [
  // --- LEVELS 1-10: BASICS ---
  createAdaptiveWord('r1', 1, {
    [Difficulty.PROFOUND]: { w: 'Up', s: 'Look up.' },
    [Difficulty.SEVERE]: { w: 'Cat', s: 'The cat sits.' },
    [Difficulty.MODERATE]: { w: 'Plan', s: 'I have a plan.' },
    [Difficulty.MILD]: { w: 'Where', s: 'Where is the dog?' }
  }),
  createAdaptiveWord('r2', 2, {
    [Difficulty.PROFOUND]: { w: 'Go', s: 'We go out.' },
    [Difficulty.SEVERE]: { w: 'Dog', s: 'My dog runs.' },
    [Difficulty.MODERATE]: { w: 'Flag', s: 'See the flag.' },
    [Difficulty.MILD]: { w: 'There', s: 'It is over there.' }
  }),
  createAdaptiveWord('r3', 3, {
    [Difficulty.PROFOUND]: { w: 'Me', s: 'Look at me.' },
    [Difficulty.SEVERE]: { w: 'Sun', s: 'The sun is hot.' },
    [Difficulty.MODERATE]: { w: 'Slip', s: 'Do not slip.' },
    [Difficulty.MILD]: { w: 'Could', s: 'I could do it.' }
  }),
  createAdaptiveWord('r4', 4, {
    [Difficulty.PROFOUND]: { w: 'My', s: 'My toy is red.' },
    [Difficulty.SEVERE]: { w: 'Pig', s: 'The pig is pink.' },
    [Difficulty.MODERATE]: { w: 'Frog', s: 'The frog jumps.' },
    [Difficulty.MILD]: { w: 'Should', s: 'You should go.' }
  }),
  createAdaptiveWord('r5', 5, {
    [Difficulty.PROFOUND]: { w: 'Is', s: 'It is a ball.' },
    [Difficulty.SEVERE]: { w: 'Bed', s: 'Go to bed.' },
    [Difficulty.MODERATE]: { w: 'Step', s: 'Take a step.' },
    [Difficulty.MILD]: { w: 'Would', s: 'I would like that.' }
  }),
  createAdaptiveWord('r6', 6, {
    [Difficulty.PROFOUND]: { w: 'On', s: 'Sit on it.' },
    [Difficulty.SEVERE]: { w: 'Hat', s: 'Wear a hat.' },
    [Difficulty.MODERATE]: { w: 'Trip', s: 'A fun trip.' },
    [Difficulty.MILD]: { w: 'Right', s: 'Turn right here.' }
  }),
  createAdaptiveWord('r7', 7, {
    [Difficulty.PROFOUND]: { w: 'At', s: 'Look at that.' },
    [Difficulty.SEVERE]: { w: 'Box', s: 'Open the box.' },
    [Difficulty.MODERATE]: { w: 'Club', s: 'Join the club.' },
    [Difficulty.MILD]: { w: 'Light', s: 'Turn on the light.' }
  }),
  createAdaptiveWord('r8', 8, {
    [Difficulty.PROFOUND]: { w: 'No', s: 'Say no.' },
    [Difficulty.SEVERE]: { w: 'Run', s: 'We run fast.' },
    [Difficulty.MODERATE]: { w: 'Drum', s: 'Hit the drum.' },
    [Difficulty.MILD]: { w: 'Night', s: 'Good night.' }
  }),
  createAdaptiveWord('r9', 9, {
    [Difficulty.PROFOUND]: { w: 'He', s: 'He is tall.' },
    [Difficulty.SEVERE]: { w: 'Sit', s: 'Sit down now.' },
    [Difficulty.MODERATE]: { w: 'Swim', s: 'I can swim.' },
    [Difficulty.MILD]: { w: 'Sight', s: 'A pretty sight.' }
  }),
  createAdaptiveWord('r10', 10, {
    [Difficulty.PROFOUND]: { w: 'Do', s: 'Do it now.' },
    [Difficulty.SEVERE]: { w: 'Map', s: 'Read the map.' },
    [Difficulty.MODERATE]: { w: 'Crab', s: 'A red crab.' },
    [Difficulty.MILD]: { w: 'Might', s: 'I might go.' }
  }),

  // --- LEVELS 11-20: INTERMEDIATE ---
  createAdaptiveWord('r11', 11, {
    [Difficulty.PROFOUND]: { w: 'Big', s: 'A big cat.' },
    [Difficulty.SEVERE]: { w: 'Star', s: 'A bright star.' },
    [Difficulty.MODERATE]: { w: 'Ship', s: 'A big ship.' },
    [Difficulty.MILD]: { w: 'Cake', s: 'Eat the cake.' }
  }),
  createAdaptiveWord('r12', 12, {
    [Difficulty.PROFOUND]: { w: 'Hot', s: 'It is hot.' },
    [Difficulty.SEVERE]: { w: 'Blue', s: 'Sky is blue.' },
    [Difficulty.MODERATE]: { w: 'Chip', s: 'Eat a chip.' },
    [Difficulty.MILD]: { w: 'Bike', s: 'Ride a bike.' }
  }),
  createAdaptiveWord('r13', 13, {
    [Difficulty.PROFOUND]: { w: 'Red', s: 'A red apple.' },
    [Difficulty.SEVERE]: { w: 'Tree', s: 'Climb the tree.' },
    [Difficulty.MODERATE]: { w: 'Fish', s: 'Fish can swim.' },
    [Difficulty.MILD]: { w: 'Home', s: 'Go back home.' }
  }),
  createAdaptiveWord('r14', 14, {
    [Difficulty.PROFOUND]: { w: 'Sad', s: 'Do not be sad.' },
    [Difficulty.SEVERE]: { w: 'Play', s: 'Let us play.' },
    [Difficulty.MODERATE]: { w: 'Thin', s: 'The paper is thin.' },
    [Difficulty.MILD]: { w: 'Cute', s: 'The puppy is cute.' }
  }),
  createAdaptiveWord('r15', 15, {
    [Difficulty.PROFOUND]: { w: 'Mad', s: 'Are you mad?' },
    [Difficulty.SEVERE]: { w: 'Snow', s: 'Cold snow.' },
    [Difficulty.MODERATE]: { w: 'Chop', s: 'Chop the wood.' },
    [Difficulty.MILD]: { w: 'Kite', s: 'Fly a kite.' }
  }),
  createAdaptiveWord('r16', 16, {
    [Difficulty.PROFOUND]: { w: 'Fun', s: 'This is fun.' },
    [Difficulty.SEVERE]: { w: 'Jump', s: 'Jump up high.' },
    [Difficulty.MODERATE]: { w: 'Shed', s: 'Tools in the shed.' },
    [Difficulty.MILD]: { w: 'Rope', s: 'Pull the rope.' }
  }),
  createAdaptiveWord('r17', 17, {
    [Difficulty.PROFOUND]: { w: 'Wet', s: 'Water is wet.' },
    [Difficulty.SEVERE]: { w: 'Milk', s: 'Drink your milk.' },
    [Difficulty.MODERATE]: { w: 'Whale', s: 'A big whale.' },
    [Difficulty.MILD]: { w: 'Note', s: 'Write a note.' }
  }),
  createAdaptiveWord('r18', 18, {
    [Difficulty.PROFOUND]: { w: 'Bus', s: 'Ride the bus.' },
    [Difficulty.SEVERE]: { w: 'Fast', s: 'Run very fast.' },
    [Difficulty.MODERATE]: { w: 'Phone', s: 'Call on phone.' },
    [Difficulty.MILD]: { w: 'Tube', s: 'A long tube.' }
  }),
  createAdaptiveWord('r19', 19, {
    [Difficulty.PROFOUND]: { w: 'Fox', s: 'See the fox.' },
    [Difficulty.SEVERE]: { w: 'Best', s: 'You are best.' },
    [Difficulty.MODERATE]: { w: 'White', s: 'White snow.' },
    [Difficulty.MILD]: { w: 'Made', s: 'I made this.' }
  }),
  createAdaptiveWord('r20', 20, {
    [Difficulty.PROFOUND]: { w: 'Bug', s: 'A little bug.' },
    [Difficulty.SEVERE]: { w: 'Stop', s: 'Please stop.' },
    [Difficulty.MODERATE]: { w: 'Think', s: 'Think about it.' },
    [Difficulty.MILD]: { w: 'Same', s: 'We are the same.' }
  }),

  // --- LEVELS 21-30: ADVANCED ---
  createAdaptiveWord('r21', 21, {
    [Difficulty.PROFOUND]: { w: 'Ball', s: 'Throw the ball.' },
    [Difficulty.SEVERE]: { w: 'Make', s: 'Make a cake.' },
    [Difficulty.MODERATE]: { w: 'Happy', s: 'I am happy.' },
    [Difficulty.MILD]: { w: 'Splash', s: 'Splash in water.' }
  }),
  createAdaptiveWord('r22', 22, {
    [Difficulty.PROFOUND]: { w: 'Car', s: 'Drive the car.' },
    [Difficulty.SEVERE]: { w: 'Ride', s: 'Ride the bus.' },
    [Difficulty.MODERATE]: { w: 'Funny', s: 'A funny joke.' },
    [Difficulty.MILD]: { w: 'Spring', s: 'Spring is here.' }
  }),
  createAdaptiveWord('r23', 23, {
    [Difficulty.PROFOUND]: { w: 'Doll', s: 'My pretty doll.' },
    [Difficulty.SEVERE]: { w: 'Like', s: 'I like you.' },
    [Difficulty.MODERATE]: { w: 'Party', s: 'Birthday party.' },
    [Difficulty.MILD]: { w: 'String', s: 'Tie the string.' }
  }),
  createAdaptiveWord('r24', 24, {
    [Difficulty.PROFOUND]: { w: 'Book', s: 'Read a book.' },
    [Difficulty.SEVERE]: { w: 'Time', s: 'What time is it?' },
    [Difficulty.MODERATE]: { w: 'Puppy', s: 'Cute puppy.' },
    [Difficulty.MILD]: { w: 'Strong', s: 'He is strong.' }
  }),
  createAdaptiveWord('r25', 25, {
    [Difficulty.PROFOUND]: { w: 'Duck', s: 'Quack says duck.' },
    [Difficulty.SEVERE]: { w: 'Name', s: 'Say your name.' },
    [Difficulty.MODERATE]: { w: 'Kitty', s: 'Soft kitty.' },
    [Difficulty.MILD]: { w: 'Screen', s: 'Watch the screen.' }
  }),
  createAdaptiveWord('r26', 26, {
    [Difficulty.PROFOUND]: { w: 'Fish', s: 'Fish swim.' },
    [Difficulty.SEVERE]: { w: 'Gate', s: 'Open the gate.' },
    [Difficulty.MODERATE]: { w: 'Penny', s: 'Save a penny.' },
    [Difficulty.MILD]: { w: 'Stream', s: 'By the stream.' }
  }),
  createAdaptiveWord('r27', 27, {
    [Difficulty.PROFOUND]: { w: 'Milk', s: 'Cold milk.' },
    [Difficulty.SEVERE]: { w: 'Bone', s: 'Dog has a bone.' },
    [Difficulty.MODERATE]: { w: 'Bunny', s: 'Hop like bunny.' },
    [Difficulty.MILD]: { w: 'Throat', s: 'Sore throat.' }
  }),
  createAdaptiveWord('r28', 28, {
    [Difficulty.PROFOUND]: { w: 'Nest', s: 'Bird in nest.' },
    [Difficulty.SEVERE]: { w: 'Five', s: 'High five.' },
    [Difficulty.MODERATE]: { w: 'Sunny', s: 'Sunny day.' },
    [Difficulty.MILD]: { w: 'Throne', s: 'King on throne.' }
  }),
  createAdaptiveWord('r29', 29, {
    [Difficulty.PROFOUND]: { w: 'Tent', s: 'Sleep in tent.' },
    [Difficulty.SEVERE]: { w: 'Nose', s: 'Touch your nose.' },
    [Difficulty.MODERATE]: { w: 'Baby', s: 'Sweet baby.' },
    [Difficulty.MILD]: { w: 'Shrimp', s: 'Tiny shrimp.' }
  }),
  createAdaptiveWord('r30', 30, {
    [Difficulty.PROFOUND]: { w: 'Vest', s: 'Wear a vest.' },
    [Difficulty.SEVERE]: { w: 'Use', s: 'Use a pen.' },
    [Difficulty.MODERATE]: { w: 'Lady', s: 'Kind lady.' },
    [Difficulty.MILD]: { w: 'Scrape', s: 'Scrape my knee.' }
  }),

  // --- LEVELS 31-40: CHALLENGE ---
  createAdaptiveWord('r31', 31, {
    [Difficulty.PROFOUND]: { w: 'Walk', s: 'We walk.' },
    [Difficulty.SEVERE]: { w: 'Inside', s: 'Go inside.' },
    [Difficulty.MODERATE]: { w: 'Banana', s: 'Yellow banana.' },
    [Difficulty.MILD]: { w: 'Science', s: 'Science class.' }
  }),
  createAdaptiveWord('r32', 32, {
    [Difficulty.PROFOUND]: { w: 'Talk', s: 'We talk.' },
    [Difficulty.SEVERE]: { w: 'Outside', s: 'Play outside.' },
    [Difficulty.MODERATE]: { w: 'Animal', s: 'Wild animal.' },
    [Difficulty.MILD]: { w: 'Friend', s: 'Best friend.' }
  }),
  createAdaptiveWord('r33', 33, {
    [Difficulty.PROFOUND]: { w: 'Sing', s: 'Sing a song.' },
    [Difficulty.SEVERE]: { w: 'Pancake', s: 'Yummy pancake.' },
    [Difficulty.MODERATE]: { w: 'Tomato', s: 'Red tomato.' },
    [Difficulty.MILD]: { w: 'School', s: 'Go to school.' }
  }),
  createAdaptiveWord('r34', 34, {
    [Difficulty.PROFOUND]: { w: 'Ring', s: 'Ring the bell.' },
    [Difficulty.SEVERE]: { w: 'Cowboy', s: 'Ride cowboy.' },
    [Difficulty.MODERATE]: { w: 'Potato', s: 'Baked potato.' },
    [Difficulty.MILD]: { w: 'People', s: 'Many people.' }
  }),
  createAdaptiveWord('r35', 35, {
    [Difficulty.PROFOUND]: { w: 'King', s: 'The king.' },
    [Difficulty.SEVERE]: { w: 'Sunset', s: 'Pretty sunset.' },
    [Difficulty.MODERATE]: { w: 'Elephant', s: 'Big elephant.' },
    [Difficulty.MILD]: { w: 'Enough', s: 'That is enough.' }
  }),
  createAdaptiveWord('r36', 36, {
    [Difficulty.PROFOUND]: { w: 'Long', s: 'Long hair.' },
    [Difficulty.SEVERE]: { w: 'Cupcake', s: 'Sweet cupcake.' },
    [Difficulty.MODERATE]: { w: 'Umbrella', s: 'Use umbrella.' },
    [Difficulty.MILD]: { w: 'Thought', s: 'I thought so.' }
  }),
  createAdaptiveWord('r37', 37, {
    [Difficulty.PROFOUND]: { w: 'Song', s: 'Loud song.' },
    [Difficulty.SEVERE]: { w: 'Backpack', s: 'School backpack.' },
    [Difficulty.MODERATE]: { w: 'Octopus', s: 'Sea octopus.' },
    [Difficulty.MILD]: { w: 'Through', s: 'Go through.' }
  }),
  createAdaptiveWord('r38', 38, {
    [Difficulty.PROFOUND]: { w: 'Bank', s: 'Piggy bank.' },
    [Difficulty.SEVERE]: { w: 'Snowman', s: 'Cold snowman.' },
    [Difficulty.MODERATE]: { w: 'Computer', s: 'Use computer.' },
    [Difficulty.MILD]: { w: 'Laugh', s: 'Laugh loud.' }
  }),
  createAdaptiveWord('r39', 39, {
    [Difficulty.PROFOUND]: { w: 'Pink', s: 'Pink dress.' },
    [Difficulty.SEVERE]: { w: 'Raincoat', s: 'Wear raincoat.' },
    [Difficulty.MODERATE]: { w: 'Hospital', s: 'The hospital.' },
    [Difficulty.MILD]: { w: 'Rough', s: 'Rough road.' }
  }),
  createAdaptiveWord('r40', 40, {
    [Difficulty.PROFOUND]: { w: 'Sink', s: 'Wash in sink.' },
    [Difficulty.SEVERE]: { w: 'Starfish', s: 'See starfish.' },
    [Difficulty.MODERATE]: { w: 'Dinosaur', s: 'Big dinosaur.' },
    [Difficulty.MILD]: { w: 'Daughter', s: 'Her daughter.' }
  }),
];

// ==========================================
// SPELLING LEVELS
// ==========================================
const makeSpell = (i: number, w: string, h: string, s?: string) => ({ 
    id: `s${i}`, 
    word: w, 
    scrambled: [], 
    hint: h,
    contextSentence: s || `The word is ${w}.` 
});

export const SPELLING_ITEMS: SpellingItem[] = [
  makeSpell(1, 'CAT', 'Meow!', 'The cat sat on the mat.'), 
  makeSpell(2, 'DOG', 'Woof!', 'The dog chased the ball.'), 
  makeSpell(3, 'BAT', 'Flying mammal', 'The bat flew at night.'), 
  makeSpell(4, 'PIG', 'Oink!', 'The pig played in the mud.'), 
  makeSpell(5, 'BUS', 'Go to school', 'The yellow bus stops here.'),
  makeSpell(6, 'NET', 'Catch butterflies', 'Catch the fish in the net.'), 
  makeSpell(7, 'TOP', 'Spinning toy', 'Spin the top on the table.'), 
  makeSpell(8, 'SUN', 'In the sky', 'The sun is hot today.'), 
  makeSpell(9, 'MOP', 'Clean the floor', 'Use the mop to clean up.'), 
  makeSpell(10, 'HAT', 'Wear on head', 'Put on your hat.'),
  
  makeSpell(11, 'FROG', 'Green jumper', 'The frog jumped in the pond.'), 
  makeSpell(12, 'CRAB', 'Walks sideways', 'The crab has claws.'), 
  makeSpell(13, 'DRUM', 'Beat it', 'Play the drum loudly.'), 
  makeSpell(14, 'FLAG', 'Wave it', 'Wave the flag high.'), 
  makeSpell(15, 'PLAN', 'A good idea', 'I have a plan for the weekend.'),
  makeSpell(16, 'SWIM', 'In the pool', 'I like to swim in the summer.'), 
  makeSpell(17, 'STOP', 'Red sign', 'Stop at the red light.'), 
  makeSpell(18, 'TWIN', 'Look alike', 'My twin sister looks like me.'), 
  makeSpell(19, 'GRAB', 'Take it', 'Grab a slice of pizza.'), 
  makeSpell(20, 'SPOT', 'A mark', 'X marks the spot.'),

  makeSpell(21, 'FISH', 'Swims in water', 'Gold fish in a bowl.'), 
  makeSpell(22, 'SHIP', 'Big boat', 'The ship sailed across the sea.'), 
  makeSpell(23, 'CHIP', 'Potato snack', 'Eat a potato chip.'), 
  makeSpell(24, 'MOTH', 'Night butterfly', 'The moth flew to the light.'), 
  makeSpell(25, 'BATH', 'Wash up', 'Take a warm bath.'),
  makeSpell(26, 'RING', 'Wear on finger', 'She wears a gold ring.'), 
  makeSpell(27, 'KING', 'Royal ruler', 'The king wore a crown.'), 
  makeSpell(28, 'SONG', 'Sing it', 'Sing a happy song.'), 
  makeSpell(29, 'WHIP', 'Cream topping', 'Whip the cream until fluffy.'), 
  makeSpell(30, 'DUCK', 'Quack!', 'The duck swam in the pond.'),

  makeSpell(31, 'HAPPY', 'Not sad', 'I am so happy today.'), 
  makeSpell(32, 'FUNNY', 'Makes you laugh', 'That was a funny joke.'), 
  makeSpell(33, 'LITTLE', 'Small', 'A little mouse ran by.'), 
  makeSpell(34, 'RABBIT', 'Long ears', 'The rabbit hopped away.'), 
  makeSpell(35, 'SUMMER', 'Hot season', 'We go to the beach in summer.'),
  makeSpell(36, 'WINTER', 'Cold season', 'Snow falls in winter.'), 
  makeSpell(37, 'YELLOW', 'Color of sun', 'The banana is yellow.'), 
  makeSpell(38, 'PURPLE', 'Color of grapes', 'Grapes can be green or purple.'), 
  makeSpell(39, 'SCHOOL', 'Place to learn', 'We learn reading at school.'), 
  makeSpell(40, 'FRIEND', 'Playmate', 'You are my best friend.')
];

// ==========================================
// MEMORY LEVELS
// ==========================================
const makeMem = (i: number, seq: string, type: 'Numbers' | 'Letters' | 'Mixed') => ({ id: `m${i}`, sequence: seq, type });

export const MEMORY_ITEMS: MemoryItem[] = [
  makeMem(1, '1 2', 'Numbers'), makeMem(2, '5 9', 'Numbers'), makeMem(3, '3 1', 'Numbers'), makeMem(4, '8 2', 'Numbers'), makeMem(5, '4 7', 'Numbers'),
  makeMem(6, '1 2 3', 'Numbers'), makeMem(7, '5 9 1', 'Numbers'), makeMem(8, '7 4 2', 'Numbers'), makeMem(9, '9 8 1', 'Numbers'), makeMem(10, '6 3 5', 'Numbers'),
  makeMem(11, 'A B', 'Letters'), makeMem(12, 'C A', 'Letters'), makeMem(13, 'F G', 'Letters'), makeMem(14, 'T P', 'Letters'), makeMem(15, 'M N', 'Letters'),
  makeMem(16, 'A B C', 'Letters'), makeMem(17, 'D O G', 'Letters'), makeMem(18, 'C A T', 'Letters'), makeMem(19, 'X Y Z', 'Letters'), makeMem(20, 'H O P', 'Letters'),
  makeMem(21, '1 5 9 2', 'Numbers'), makeMem(22, '8 3 7 1', 'Numbers'), makeMem(23, '4 2 0 6', 'Numbers'), makeMem(24, '9 5 1 4', 'Numbers'), makeMem(25, '3 8 2 7', 'Numbers'),
  makeMem(26, '1 2 3 4 5', 'Numbers'), makeMem(27, '9 8 7 6 5', 'Numbers'), makeMem(28, '5 1 9 2 8', 'Numbers'), makeMem(29, '7 4 1 8 5', 'Numbers'), makeMem(30, '3 0 2 9 6', 'Numbers'),
  makeMem(31, 'A 1 B', 'Mixed'), makeMem(32, '1 C 2', 'Mixed'), makeMem(33, 'T 4 P', 'Mixed'), makeMem(34, '7 H 3', 'Mixed'), makeMem(35, 'A 1 B 2', 'Mixed'),
  makeMem(36, '9 Z 8 Y', 'Mixed'), makeMem(37, 'K 5 L 6', 'Mixed'), makeMem(38, '1 A 2 B 3', 'Mixed'), makeMem(39, 'X 9 Y 8 Z', 'Mixed'), makeMem(40, 'M 1 N 2 O 3', 'Mixed'),
];

export const DIFFICULTY_SETTINGS = {
  [Difficulty.MILD]: {
    strokeWidth: 35,
    tolerance: 5, // Reduced from 30
    description: "Standard guides. Gentle correction."
  },
  [Difficulty.MODERATE]: {
    strokeWidth: 45,
    tolerance: 10, // Reduced from 40
    description: "Thicker guides to help stay on track."
  },
  [Difficulty.SEVERE]: {
    strokeWidth: 60,
    tolerance: 15, // Reduced from 50
    description: "Very thick guides. High error tolerance."
  },
  [Difficulty.PROFOUND]: {
    strokeWidth: 80,
    tolerance: 30, // Reduced from 70
    description: "Maximum visual guidance. Focus on movement."
  }
};
