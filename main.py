from ciphertext_env import CipherEnv
from agent import HonestAgent, ImposterAgent
from run_episode import run_episode

env = CipherEnv(text_len = 40, max_rounds = 100)

vocab = [

# === CORE SPACE-PADDED CRIBS (WORD BOUNDARIES) ===
" THE ", " AND ", " OF ", " TO ", " IN ", " ON ", " FOR ", " WITH ", " FROM ",
" BY ", " AT ", " AS ", " IF ", " OR ", " WHEN ", " THAN ", " INTO ", " OVER ",
" UNDER ", " THROUGH ", " BETWEEN ", " ABOUT ", " AGAINST ", " WITHOUT ",

# Leading/trailing space variants
" THE", " AND", " OF", " TO", " IN", " ON", " FOR", " WITH",
"THE ", "AND ", "OF ", "TO ", "IN ", "ON ", "FOR ", "WITH ",

# === TOP FREQUENCY ENGLISH WORDS (100) ===
"THE", "BE", "AND", "OF", "A", "IN", "TO", "HAVE", "IT", "I",
"THAT", "FOR", "YOU", "HE", "WITH", "ON", "DO", "SAY", "THIS", "THEY",
"AT", "BUT", "WE", "HIS", "FROM", "NOT", "BY", "SHE", "OR", "AS",
"WHAT", "GO", "THEIR", "CAN", "WHO", "GET", "IF", "WOULD", "HER", "ALL",
"MY", "MAKE", "ABOUT", "KNOW", "WILL", "UP", "ONE", "TIME", "THERE", "YEAR",
"SO", "THINK", "WHEN", "WHICH", "THEM", "SOME", "ME", "PEOPLE",
"TAKE", "OUT", "INTO", "JUST", "SEE", "HIM", "YOUR", "COME",
"COULD", "NOW", "THAN", "LIKE", "OTHER", "HOW", "THEN", "ITS",
"TWO", "MORE", "THESE", "WANT", "WAY", "LOOK", "FIRST", "ALSO",
"NEW", "BECAUSE", "DAY", "USE", "NO", "MAN", "FIND", "HERE",
"THING", "GIVE", "MANY", "WELL",

# === HIGH VALUE N-GRAM STRUCTURES (50) ===
"ING", "TION", "ENT", "MENT", "NESS", "ABLE", "LESS", "ING ", "ED ",
"ER ", "RE ", "ST ", "AL ", "BE ", "DE ", "CH ", "EN ", "AN ", "AR ",
"OR ", "TH ", "HE ", "IN ", "ON ", "ED", "ER", "RE", "ST", "AL",
"BE", "DE", "CH", "EN", "AN", "AR", "OR",
"THE ", " HE ", " AN ", " TO ", " OF ", " IN ", " ON ", " AT ",

# === COMMON 2–4 LETTER WORDS (100) ===
"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "ANY", "CAN",
"HAD", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM",
"HIS", "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO",
"BOY", "DID", "END", "FEW", "JOB", "LET", "LIFE", "MAY", "OFF", "RUN",
"SIT", "TRY", "USE", "WAR", "BAD", "BIG", "FAR", "FUN", "HOT",
"KID", "LOW", "MAP", "NET", "PAY", "PER", "PUT", "RED", "TOO", "TOP",
"ACT", "AIM", "ARM", "ASK", "CUT", "DUE", "EAT", "EGG", "FIX", "FIT",
"GUN", "HIT", "KEY", "LAW", "LIE", "LOG", "MAD", "MEN", "ODD", "OIL",
"PAN", "POP", "RAT", "RAW", "RIP", "SEA", "SET", "SIX", "TAX", "TIP",

# === MULTIWORD PHRASES (60) ===
"IT IS ", "THERE IS ", "THERE ARE ", "THIS IS ", "THAT IS ",
"IN THE ", "ON THE ", "FOR THE ", "TO THE ", "OF THE ",
"AS A ", "AS THE ", "IN A ", "WITH THE ", "BY THE ",
"END OF ", "PART OF ", "ONE OF ", "MOST OF ",
"FIRST OF ", "SOME OF ", "A LOT ", "A FEW ", "A LITTLE ",
"MORE THAN ", "LESS THAN ", "NO LONGER ", "NO MORE ",
"ON TOP ", "IN FRONT ", "IN ORDER ", "SUCH AS ",
"KIND OF ", "SORT OF ", "OUT OF ", "BACK TO ",
"ABLE TO ", "ABOUT TO ", "GOING TO ", "SUPPOSED TO ",
" USED TO ", " HAVE BEEN ", " WOULD BE ", " COULD BE ",
" SHOULD BE ", " MUST BE ", " MIGHT BE ",
"LOOKING FOR ", "LOOKING AT ", "THINKING OF ", "THINKING ABOUT ",
"ACCORDING TO ", "DUE TO ", "BASED ON ", "DEPENDING ON ",

# === EXTENDED HIGH-FREQUENCY WORDS (190+) ===
# (Large block of common Brown corpus words)
"AGAIN", "ALWAYS", "AMERICAN", "ANOTHER", "AROUND", "BECAME", "BEFORE",
"BEHIND", "BEYOND", "BETWEEN", "BETTER", "BOTH", "BRING", "BUSINESS",
"CHILDREN", "COMPANY", "COUNTRY", "COURSE", "DEVELOPMENT", "DIFFERENT",
"DURING", "EARLY", "ECONOMIC", "EDUCATION", "ENGLISH", "ENOUGH", "EVERY",
"EXAMPLE", "FEDERAL", "FOLLOWING", "GENERAL", "GOVERNMENT", "GROUP",
"HAPPENED", "IMPORTANT", "INTEREST", "INVOLVED", "KNOWLEDGE", "LARGER",
"LEARNED", "LEARNING", "LEAST", "LEAVE", "LEFT", "LITTLE", "LIVING",
"LONGER", "MIGHT", "MONEY", "MONTH", "MOTHER", "NATION", "NEVER", "NEXT",
"NUMBER", "OCCURRED", "OFFER", "ORDER", "PERHAPS", "PLACE", "POINT",
"PRESENT", "PROBABLY", "PROVIDE", "PUBLIC", "QUESTION", "QUITE", "REASON",
"REALLY", "RECEIVED", "RESEARCH", "RESULT", "SEVERAL", "SCHOOL",
"SOCIETY", "SYSTEM", "THOUGHT", "TOWARD", "TRUE", "UNIVERSITY", "UNTIL",
"UPON", "VARIOUS", "WEEK", "WHILE", "WORLD", "WOULD", "WORK", "WRITTEN",
"YOUNG", "ACROSS", "ALREADY", "ALTHOUGH", "ANYTHING", "BEGAN", "BESIDE",
"CERTAIN", "CHANGED", "CHANGES", "CLEAR", "CLOSE", "COMMON", "COMMUNITY",
"CONSIDER", "CONTINUE", "CONTROL", "DURING", "EFFECT", "EFFORT",
"EVER", "FACT", "FAMILY", "FEEL", "FIGURE", "FOLLOW", "FORM", "FREE",
"FRIEND", "FULL", "FURTHER", "GREAT", "GROUP", "HARD", "HEAD", "HELD",
"HIGH", "HISTORY", "HUMAN", "IDEA", "INCREASE", "INDIVIDUAL", "LEAD",
"LEVEL", "LIGHT", "LINE", "MAIN", "MAJOR", "MANAGEMENT", "MATERIAL",
"MEAN", "METHOD", "MIDDLE", "MODEL", "MODERN", "MOMENT", "NATURAL",
"NECESSARY", "NEED", "NIGHT", "NORTH", "OFTEN", "ONCE", "OPEN", "OPERATION",
"OTHERWISE", "PAPER", "PART", "PAST", "PATTERN", "PERIOD", "PERSON",
"PIECE", "PLACE", "PLANT", "POWER", "PROBLEM", "PROCESS", "PROGRAM",
"PROJECT", "QUALITY", "QUESTION", "RANGE", "RATE", "READ", "READY",
"REAL", "RECORD", "REPORT", "RIGHT", "ROLE", "ROOM", "SERVICE", "SIDE",
"SIMPLE", "SINGLE", "SOCIAL", "SPECIAL", "SQUARE", "STORY", "STUDY",
"SUPPORT", "TAKEN", "TEACHER", "TERMS", "THING", "THIRD", "THOSE",
"TODAY", "TRADE", "TYPE", "VALUE", "WATCH", "WEST", "WHOLE", "WOMEN",
"WONDER", "WORD", "WORKERS", "WRITING",

]


agents = [
    HonestAgent("A1", vocab, env.text_len, env),
    HonestAgent("A2", vocab, env.text_len, env),
    ImposterAgent("IMP", vocab, env.text_len, env)
]

if __name__ == "__main__":
    num_episodes = 200
    completion_history = []
    for ep in range(num_episodes):
        stats = run_episode(env, agents)
        completion = stats.get("completion", 0.0)
        print(f"Episode {ep+1}: completion = {completion*100:.2f}%")
        completion_history.append(completion)
    
    print(f"Completion Rate History: {completion_history}")
