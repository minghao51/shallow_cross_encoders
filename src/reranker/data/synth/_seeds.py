"""Default seed data for synthetic pair generation across domains."""

from __future__ import annotations

# ruff: noqa: E501

DEFAULT_PAIR_SEEDS = [
    # Python / Programming
    {
        "query": "python dataclass default factory",
        "domain": "python",
        "variations": [
            "how to set default factory in python dataclass",
            "avoid shared mutable defaults dataclass field",
            "python dataclass mutable default argument workaround",
            "default_factory list dict dataclass example",
        ],
        "positive": (
            "Use field(default_factory=list) to avoid shared mutable defaults in dataclasses."
        ),
        "negative": "JavaScript arrays can be cloned with the spread operator.",
    },
    {
        "query": "python async await event loop",
        "domain": "python",
        "variations": [
            "how does python asyncio event loop work",
            "python coroutine scheduling cooperative multitasking",
            "understanding async await in python 3",
            "python concurrent I/O without threads asyncio",
        ],
        "positive": (
            "The asyncio event loop schedules and runs coroutines cooperatively, enabling"
            " concurrent I/O without threads."
        ),
        "negative": "Python list comprehensions provide a concise way to create new lists.",
    },
    {
        "query": "python context manager with statement",
        "domain": "python",
        "variations": [
            "python __enter__ __exit__ resource management",
            "how to create custom context manager python",
            "with statement resource lifecycle python",
            "contextlib context manager decorator python",
        ],
        "positive": (
            "A context manager implements __enter__ and __exit__ to manage resource lifecycle"
            " such as file handles or database connections."
        ),
        "negative": "Python decorators wrap functions to add behavior without modifying source code.",
    },
    # Information Retrieval
    {
        "query": "bm25 exact term match retrieval",
        "domain": "information_retrieval",
        "variations": [
            "how does BM25 ranking algorithm work",
            "BM25 term frequency document length normalization",
            "okapi BM25 vs vector space model",
            "BM25 scoring formula explained",
        ],
        "positive": (
            "BM25 emphasizes exact term overlap and document length normalization for ranking."
        ),
        "negative": "Dense embeddings are usually trained with contrastive objectives.",
    },
    {
        "query": "tf-idf term weighting scheme",
        "domain": "information_retrieval",
        "variations": [
            "how tf-idf calculates term importance",
            "inverse document frequency formula explanation",
            "why rare terms score higher in tf-idf",
            "tf-idf vs raw term frequency comparison",
        ],
        "positive": (
            "TF-IDF multiplies term frequency by inverse document frequency to down-weight"
            " common words and highlight distinctive terms."
        ),
        "negative": "Neural networks learn representations through gradient-based optimization.",
    },
    {
        "query": "inverted index search engine structure",
        "domain": "information_retrieval",
        "variations": [
            "how does an inverted index work in search",
            "postings list document ID mapping search engines",
            "inverted index vs forward index difference",
            "building a search index from scratch",
        ],
        "positive": (
            "An inverted index maps each term to a postings list of document IDs, enabling"
            " fast lookup of documents containing query terms."
        ),
        "negative": "Relational databases use B-tree indexes for range queries on numeric columns.",
    },
    # Singapore Real Estate
    {
        "query": "singapore hdb school proximity value",
        "domain": "singapore_real_estate",
        "variations": [
            "does living near good schools increase HDB resale price",
            "Singapore property value near primary schools",
            "HDB flat proximity to schools impact on demand",
            "school distance effect on Singapore housing prices",
        ],
        "positive": (
            "Properties near sought-after primary schools often attract stronger demand in"
            " Singapore."
        ),
        "negative": "Ocean currents influence monsoon weather patterns across the region.",
    },
    # Machine Learning
    {
        "query": "transformer attention mechanism explained",
        "domain": "machine_learning",
        "variations": [
            "how does self-attention work in transformers",
            "query key value attention mechanism NLP",
            "scaled dot-product attention formula",
            "why transformers use multi-head attention",
        ],
        "positive": (
            "Self-attention computes weighted sums of value vectors where weights come from"
            " dot-product similarity between query and key projections."
        ),
        "negative": "Convolutional neural networks apply sliding filters to detect spatial patterns.",
    },
    {
        "query": "gradient descent learning rate tuning",
        "domain": "machine_learning",
        "variations": [
            "how to choose learning rate for gradient descent",
            "learning rate too high too low effects",
            "learning rate scheduling strategies deep learning",
            "adaptive learning rate optimizers Adam SGD",
        ],
        "positive": (
            "A learning rate that is too large causes oscillation around minima, while one that"
            " is too small results in slow convergence."
        ),
        "negative": "Decision trees split on features that maximize information gain at each node.",
    },
    {
        "query": "overfitting regularization techniques",
        "domain": "machine_learning",
        "variations": [
            "L1 vs L2 regularization difference",
            "how to prevent overfitting in machine learning models",
            "weight decay penalty neural network training",
            "dropout regularization explained",
        ],
        "positive": (
            "L2 regularization adds the squared magnitude of weights to the loss function,"
            " penalizing large parameters and reducing overfitting."
        ),
        "negative": "Ensemble methods combine multiple weak learners to improve predictive accuracy.",
    },
    # DevOps / Infrastructure
    {
        "query": "kubernetes pod restart policy always",
        "domain": "devops",
        "variations": [
            "kubernetes restartPolicy options explained",
            "when to use Always restart policy k8s",
            "k8s pod crash loop backoff behavior",
            "container restart policy OnFailure vs Always",
        ],
        "positive": (
            "The Always restart policy ensures that containers are restarted regardless of the"
            " exit code, maintaining service availability."
        ),
        "negative": "Docker volumes persist data across container lifecycle events.",
    },
    {
        "query": "ci cd pipeline stages deployment",
        "domain": "devops",
        "variations": [
            "continuous integration delivery pipeline stages",
            "automated deployment pipeline best practices",
            "CI CD build test deploy workflow",
            "staging to production deployment strategies",
        ],
        "positive": (
            "A typical CI/CD pipeline includes build, test, security scan, staging deployment,"
            " and production rollout stages with automated rollback."
        ),
        "negative": "Agile sprints organize development work into fixed-length iterations.",
    },
    # Finance
    {
        "query": "compound interest vs simple interest formula",
        "domain": "finance",
        "variations": [
            "difference between compound and simple interest",
            "how compound interest grows over time formula",
            "simple interest calculation example",
            "why compound interest earns more than simple",
        ],
        "positive": (
            "Compound interest applies to accumulated principal plus earned interest, while simple"
            " interest is calculated only on the original principal amount."
        ),
        "negative": "Stock market indices track the performance of a basket of selected equities.",
    },
    {
        "query": "net present value investment decision",
        "domain": "finance",
        "variations": [
            "how to calculate NPV for investment projects",
            "discount rate in net present value analysis",
            "positive NPV means good investment",
            "NPV vs IRR capital budgeting comparison",
        ],
        "positive": (
            "NPV discounts future cash flows to present value using a required rate of return;"
            " positive NPV indicates a value-creating investment."
        ),
        "negative": "Technical analysis uses chart patterns to predict short-term price movements.",
    },
    # Legal
    {
        "query": "statute of limitations breach of contract",
        "domain": "legal",
        "variations": [
            "how long do you have to sue for breach of contract",
            "contract claim filing deadline time limit",
            "statute of limitations contract law by jurisdiction",
            "when does statute of limitations start for contracts",
        ],
        "positive": (
            "The statute of limitations sets a deadline for filing a breach of contract claim,"
            " typically ranging from three to six years depending on jurisdiction."
        ),
        "negative": "Mediation is a voluntary process where a neutral third party facilitates negotiation.",
    },
    {
        "query": "intellectual property fair use doctrine",
        "domain": "legal",
        "variations": [
            "what qualifies as fair use copyright law",
            "fair use four factors explained",
            "how much of copyrighted material can you use",
            "fair use vs public domain difference",
        ],
        "positive": (
            "Fair use permits limited use of copyrighted material for purposes such as criticism,"
            " commentary, news reporting, and research without permission."
        ),
        "negative": "Patents protect inventions for twenty years from the filing date.",
    },
    # Education
    {
        "query": "bloom taxonomy cognitive levels hierarchy",
        "domain": "education",
        "variations": [
            "six levels of Bloom's Taxonomy explained",
            "Bloom's Taxonomy from remembering to creating",
            "higher order thinking skills Bloom framework",
            "applying Bloom's Taxonomy in lesson planning",
        ],
        "positive": (
            "Bloom's Taxonomy orders cognitive skills from remembering and understanding through"
            " applying, analyzing, evaluating, and creating."
        ),
        "negative": "Standardized tests measure student performance against a common benchmark.",
    },
    {
        "query": "spaced repetition learning retention",
        "domain": "education",
        "variations": [
            "how spaced repetition improves memory retention",
            "spacing effect learning science evidence",
            "optimal review intervals for long-term memory",
            "Anki flashcard spaced repetition algorithm",
        ],
        "positive": (
            "Spaced repetition schedules reviews at increasing intervals, exploiting the spacing"
            " effect to strengthen long-term memory retention."
        ),
        "negative": "Active learning engages students through discussion and problem-solving activities.",
    },
    # Healthcare
    {
        "query": "type 1 vs type 2 diabetes difference",
        "domain": "healthcare",
        "variations": [
            "difference between type 1 and type 2 diabetes",
            "autoimmune diabetes vs insulin resistance",
            "type 1 diabetes causes symptoms treatment",
            "can type 2 diabetes become insulin dependent",
        ],
        "positive": (
            "Type 1 diabetes is an autoimmune condition destroying insulin-producing cells,"
            " while type 2 involves insulin resistance and relative deficiency."
        ),
        "negative": "Cardiovascular exercise strengthens the heart muscle and improves circulation.",
    },
    # Climate / Science
    {
        "query": "greenhouse effect vs global warming",
        "domain": "climate_science",
        "variations": [
            "difference between greenhouse effect and global warming",
            "is the greenhouse effect natural or human caused",
            "how greenhouse gases trap heat in atmosphere",
            "global warming causes and evidence",
        ],
        "positive": (
            "The greenhouse effect is a natural process trapping heat in the atmosphere, while"
            " global warming refers to the human-amplified increase in average temperatures."
        ),
        "negative": "Plate tectonics explains the movement of Earth's lithospheric plates.",
    },
    # Web Development
    {
        "query": "react useEffect cleanup function purpose",
        "domain": "web_development",
        "variations": [
            "why does useEffect need cleanup function",
            "react useEffect return function memory leak",
            "how to clean up subscriptions in React hooks",
            "useEffect componentWillUnmount equivalent",
        ],
        "positive": (
            "The cleanup function returned from useEffect runs before the component unmounts or"
            " before the effect re-runs, preventing memory leaks from subscriptions or timers."
        ),
        "negative": "CSS Grid layout divides a page into rows and columns for two-dimensional positioning.",
    },
    {
        "query": "rest api idempotent methods",
        "domain": "web_development",
        "variations": [
            "which HTTP methods are idempotent",
            "why PUT and DELETE are idempotent but POST is not",
            "idempotency in REST API design",
            "safe retry behavior HTTP methods",
        ],
        "positive": (
            "Idempotent HTTP methods like PUT and DELETE produce the same result regardless of"
            " how many times they are called, enabling safe retries."
        ),
        "negative": "WebSockets provide full-duplex communication channels over a single TCP connection.",
    },
    # Data Engineering
    {
        "query": "apache spark lazy evaluation advantage",
        "domain": "data_engineering",
        "variations": [
            "why does Spark use lazy evaluation",
            "Spark DAG execution plan optimization",
            "transformation vs action in Apache Spark",
            "Spark lazy evaluation fault tolerance",
        ],
        "positive": (
            "Spark's lazy evaluation builds a DAG of transformations and only executes when an"
            " action is called, enabling query optimization and fault tolerance."
        ),
        "negative": "Apache Kafka streams events between distributed systems in real time.",
    },
    {
        "query": "data warehouse star schema design",
        "domain": "data_engineering",
        "variations": [
            "star schema vs snowflake schema difference",
            "fact table dimension table data warehouse",
            "why use star schema for analytics",
            "dimensional modeling best practices",
        ],
        "positive": (
            "A star schema centers a fact table surrounded by dimension tables, optimizing for"
            " analytical queries with simple joins."
        ),
        "negative": "ETL pipelines extract data from source systems, transform it, and load it into targets.",
    },
    # Security
    {
        "query": "oauth2 authorization code flow steps",
        "domain": "security",
        "variations": [
            "how OAuth2 authorization code grant works",
            "OAuth2 flow redirect exchange access token",
            "authorization code vs implicit grant OAuth2",
            "PKCE extension OAuth2 public clients",
        ],
        "positive": (
            "The authorization code flow redirects the user to the provider, returns a code to"
            " the client, which then exchanges it for an access token server-to-server."
        ),
        "negative": "Firewalls filter network traffic based on predefined security rules.",
    },
    # Mathematics
    {
        "query": "bayes theorem conditional probability",
        "domain": "mathematics",
        "variations": [
            "how to apply Bayes theorem with examples",
            "conditional probability P(A|B) vs P(B|A)",
            "Bayesian inference updating priors with evidence",
            "Bayes theorem real world applications",
        ],
        "positive": (
            "Bayes' theorem relates P(A|B) to P(B|A) through the ratio of priors, enabling"
            " updating beliefs with new evidence."
        ),
        "negative": "The central limit theorem states that sample means converge to a normal distribution.",
    },
    # Biology
    {
        "query": "mitosis vs meiosis cell division",
        "domain": "biology",
        "variations": [
            "difference between mitosis and meiosis",
            "how many cells produced mitosis meiosis",
            "haploid diploid cell division comparison",
            "stages of meiosis prophase metaphase",
        ],
        "positive": (
            "Mitosis produces two genetically identical daughter cells, while meiosis produces"
            " four genetically diverse haploid cells for sexual reproduction."
        ),
        "negative": "DNA replication occurs during the S phase of the cell cycle.",
    },
    # NLP / Search
    {
        "query": "semantic search vs keyword search",
        "domain": "nlp_search",
        "variations": [
            "difference between semantic and keyword search",
            "how vector embeddings improve search relevance",
            "dense retrieval vs sparse retrieval comparison",
            "understanding query intent in search engines",
        ],
        "positive": (
            "Semantic search understands query intent and contextual meaning using embeddings,"
            " while keyword search matches exact terms regardless of meaning."
        ),
        "negative": "PageRank ranks web pages based on the quantity and quality of inbound links.",
    },
    {
        "query": "cross-encoder vs bi-encoder reranking",
        "domain": "nlp_search",
        "variations": [
            "cross-encoder bi-encoder architecture difference",
            "when to use cross-encoder for reranking",
            "bi-encoder retrieval speed vs cross-encoder accuracy",
            "two-stage retrieval pipeline reranking",
        ],
        "positive": (
            "Cross-encoders process query and document jointly through attention for accurate"
            " scoring, while bi-encoders embed them independently for faster retrieval."
        ),
        "negative": "Word embeddings map tokens to dense vectors capturing semantic similarity.",
    },
    # Systems / Performance
    {
        "query": "caching strategies cache invalidation",
        "domain": "systems",
        "variations": [
            "cache invalidation strategies explained",
            "TTL write-through write-back cache policies",
            "how to handle stale cache data",
            "cache consistency distributed systems",
        ],
        "positive": (
            "Cache invalidation strategies include time-to-live expiration, write-through updates,"
            " and explicit purge events to maintain data consistency."
        ),
        "negative": "Load balancers distribute incoming requests across multiple backend servers.",
    },
]
