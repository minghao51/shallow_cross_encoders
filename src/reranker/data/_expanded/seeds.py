"""Expanded dataset generation with diverse queries and balanced relevance."""
# ruff: noqa E501

from __future__ import annotations

from reranker.data._expanded.types import ExpandedSeedMap

DOMAIN_SEEDS: ExpandedSeedMap = {
    "programming_python": [
        {
            "query": "python dataclass default factory",
            "docs": {
                3: "Use field(default_factory=list) to avoid shared mutable defaults in Python dataclasses. This pattern prevents bugs when mutable objects like lists or dicts are used as default values.",
                2: "Python dataclasses provide the dataclass decorator which automatically generates __init__, __repr__, and other dunder methods. You can specify default values for fields.",
                1: "Python classes can define __init__ methods to initialize instance attributes. The __init__ method is called when a new instance is created.",
                0: "Python decorators like @property can create computed attributes in classes, but they don't affect __init__ parameter defaults or field initialization.",
            },
        },
        {
            "query": "asyncio gather vs create_task",
            "docs": {
                3: "asyncio.gather() runs multiple coroutines concurrently and returns results in order. asyncio.create_task() schedules a single coroutine. Use gather for all results, create_task for individual control.",
                2: "The asyncio module provides infrastructure for writing concurrent code using coroutines. Tasks schedule coroutines for execution in the event loop.",
                1: "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming. The language has a large standard library.",
                0: "Python threading module provides Lock and RLock for thread synchronization, but the GIL limits true parallelism for CPU-bound tasks.",
            },
        },
        {
            "query": "python type hints generic container",
            "docs": {
                3: "Use TypeVar and Generic from the typing module to create generic container classes. class Box(Generic[T]) enables type checkers to verify correct usage of generic types.",
                2: "Python 3.5+ supports type hints through the typing module. Generic types like List[int], Dict[str, Any] allow specifying container element types.",
                1: "Python's type system is dynamically typed at runtime. Type hints are not enforced by the interpreter but can be checked by tools like mypy.",
                0: "Python duck typing means objects are valid if they have the right methods, regardless of their declared type or class hierarchy.",
            },
        },
        {
            "query": "python context manager with statement",
            "docs": {
                3: "Implement __enter__ and __exit__ methods for context managers. Use contextlib.contextmanager decorator with yield for simpler cases. The with statement ensures cleanup even on exceptions.",
                2: "Python context managers handle resource setup and teardown. The with statement calls __enter__ on entry and __exit__ on exit, even if exceptions occur.",
                1: "Python try/finally blocks ensure cleanup code runs regardless of exceptions. This is the manual approach before context managers existed.",
                0: "Python garbage collection uses reference counting with a cyclic garbage collector. Objects are destroyed when reference count reaches zero.",
            },
        },
        {
            "query": "python metaclass vs class decorator",
            "docs": {
                3: "Metaclasses control class creation via __new__ on the metaclass. Class decorators modify classes after creation. Use decorators for simple modifications, metaclasses for complex class generation.",
                2: "Python metaclasses define how classes behave. type is the default metaclass. Custom metaclasses can intercept class creation, modify attributes, or enforce patterns.",
                1: "Python inheritance allows subclasses to extend parent class behavior. Method resolution order (MRO) determines which method is called.",
                0: "Python modules are singletons imported once and cached in sys.modules. Subsequent imports return the cached module object.",
            },
        },
        {
            "query": "python descriptor protocol property",
            "docs": {
                3: "Descriptors implement __get__, __set__, or __delete__. Properties are built on descriptors. Use descriptors for reusable attribute-level logic across multiple classes.",
                2: "Python's property() builtin creates managed attributes with getter, setter, and deleter. The @property decorator syntax provides a clean way to define computed attributes.",
                1: "Python attributes can be instance-level or class-level. Class attributes are shared across all instances unless overridden.",
                0: "Python __slots__ restricts instance attributes to a predefined set, reducing memory usage by preventing __dict__ creation.",
            },
        },
        {
            "query": "python itertools chain product combinations",
            "docs": {
                3: "itertools.chain() flattens iterables. product() computes Cartesian products. combinations() generates r-length subsequences. permutations() generates all orderings. All are lazy and memory-efficient.",
                2: "The itertools module provides fast, memory-efficient tools for creating iterators. Functions like chain, cycle, and repeat compose well for complex iteration patterns.",
                1: "Python generators use yield to produce values lazily. Generator expressions provide concise syntax similar to list comprehensions.",
                0: "Python list comprehensions create lists in a single expression. They're faster than equivalent for loops but materialize all results in memory.",
            },
        },
    ],
    "programming_web": [
        {
            "query": "react useEffect cleanup function",
            "docs": {
                3: "The useEffect cleanup function runs when the component unmounts or before re-running. Return a function to clean up subscriptions, timers, or event listeners to prevent memory leaks.",
                2: "React useEffect hook handles side effects in function components. It runs after render and can optionally clean up. The dependency array controls re-execution.",
                1: "React components can be class-based or function-based. Function components with hooks are the modern recommended approach for most use cases.",
                0: "React state updates are batched in React 18. Multiple setState calls in event handlers are grouped into a single re-render.",
            },
        },
        {
            "query": "CSS grid vs flexbox layout",
            "docs": {
                3: "CSS Grid is two-dimensional (rows AND columns) while Flexbox is one-dimensional (row OR column). Use Grid for page layouts, Flexbox for component alignment.",
                2: "CSS Grid Layout divides a page into regions defined by rows and columns. Use display: grid and grid-template-columns/rows to define structure.",
                1: "CSS positioning includes static, relative, absolute, fixed, and sticky values. Each positions elements differently relative to their normal flow.",
                0: "CSS specificity determines which rule applies when selectors conflict. Inline styles beat IDs, which beat classes, which beat elements.",
            },
        },
        {
            "query": "REST API pagination best practices",
            "docs": {
                3: "Use cursor-based pagination for large datasets. Return next_cursor in headers. Include total_count when feasible. Support page_size with defaults of 20-100 items.",
                2: "API pagination divides large result sets into pages. Approaches include offset/limit, cursor-based, and keyset pagination. Choose based on dataset size.",
                1: "RESTful APIs should use appropriate HTTP methods: GET for retrieval, POST for creation, PUT for update, DELETE for removal.",
                0: "API versioning strategies include URL path versioning, header-based versioning, and content negotiation. Each has trade-offs.",
            },
        },
        {
            "query": "react useMemo vs useCallback optimization",
            "docs": {
                3: "useMemo caches computed values, useCallback caches functions. Both prevent unnecessary recalculations. Use when computation is expensive or referential equality matters for child components.",
                2: "React performance optimization uses memo, useMemo, and useCallback to prevent unnecessary re-renders. Profile before optimizing to find real bottlenecks.",
                1: "React re-renders when state or props change. The reconciliation algorithm compares virtual DOM trees to minimize actual DOM updates.",
                0: "Server-side rendering generates HTML on the server before sending to the client. Next.js and Remix provide React SSR frameworks.",
            },
        },
        {
            "query": "webpack code splitting lazy loading",
            "docs": {
                3: "Use dynamic import() for code splitting. React.lazy() enables component-level lazy loading. Configure splitChunks in webpack to extract vendor bundles and shared code.",
                2: "Code splitting divides bundles into smaller chunks loaded on demand. Webpack supports entry point splitting, dynamic imports, and prefetch/preload hints.",
                1: "Module bundlers combine JavaScript files and dependencies into optimized bundles. They handle transpilation, minification, and asset management.",
                0: "Content Delivery Networks cache static assets at edge locations worldwide. CDNs reduce latency by serving content from geographically closer servers.",
            },
        },
        {
            "query": "web accessibility ARIA roles labels",
            "docs": {
                3: "Use semantic HTML first, then ARIA roles to fill gaps. aria-label provides accessible names. aria-live announces dynamic content changes. Never use ARIA to override native semantics.",
                2: "Web accessibility ensures all users can interact with content. ARIA roles, states, and properties enhance semantics for assistive technologies.",
                1: "Keyboard navigation requires focusable elements and visible focus indicators. Tab order follows DOM order by default.",
                0: "Web Performance APIs like IntersectionObserver enable lazy loading and infinite scroll patterns efficiently.",
            },
        },
        {
            "query": "HTTP caching headers ETag Cache-Control",
            "docs": {
                3: "Cache-Control max-age sets freshness lifetime. ETag enables conditional requests with If-None-Match. Use immutable for fingerprinted assets. no-cache validates, no-store prevents caching entirely.",
                2: "HTTP caching reduces latency by storing responses locally. Strong caching (max-age) serves without validation. Weak caching (ETag) validates with the server.",
                1: "HTTP status codes indicate response results. 200 OK, 301 permanent redirect, 304 not modified, 404 not found, 500 server error.",
                0: "WebSockets provide full-duplex communication channels over a single TCP connection for real-time applications.",
            },
        },
    ],
    "machine_learning": [
        {
            "query": "transformer attention mechanism",
            "docs": {
                3: "Self-attention computes weighted sums of value vectors where weights come from query-key dot products scaled by sqrt(d_k). Multi-head attention captures different relationship types.",
                2: "The Transformer architecture uses attention instead of recurrence. The attention function maps queries and key-value pairs to outputs, focusing on relevant input parts.",
                1: "Neural networks learn representations through layered transformations. Each layer applies a linear transformation followed by a non-linear activation like ReLU.",
                0: "Recurrent Neural Networks process sequences step by step, maintaining hidden states. LSTMs and GRUs address vanishing gradient problems in vanilla RNNs.",
            },
        },
        {
            "query": "gradient clipping training stability",
            "docs": {
                3: "Gradient clipping prevents exploding gradients by rescaling when norm exceeds a threshold. Use torch.nn.utils.clip_grad_norm_(). Essential for RNNs and deep networks.",
                2: "Training deep networks can suffer from exploding or vanishing gradients. Gradient clipping works alongside careful initialization, normalization layers, and residual connections.",
                1: "Learning rate schedules adjust step size during training. Common strategies include step decay, cosine annealing, and warmup followed by decay.",
                0: "Batch normalization normalizes layer inputs to stabilize training. It adds learnable scale and shift parameters per channel.",
            },
        },
        {
            "query": "contrastive learning data augmentation",
            "docs": {
                3: "Contrastive learning creates positive pairs through augmentations of the same image. Key augmentations include random cropping, color jittering, and Gaussian blur.",
                2: "Self-supervised contrastive learning pulls augmented views of the same sample together while pushing different samples apart. InfoNCE loss is commonly used.",
                1: "Data augmentation increases training set diversity through transformations like rotation, flipping, and cropping. It acts as a regularizer.",
                0: "Transfer learning fine-tunes pretrained models on new tasks. Feature extraction freezes the backbone while training a new classifier head.",
            },
        },
        {
            "query": "learning rate warmup scheduler",
            "docs": {
                3: "Linear warmup gradually increases learning rate from near-zero to target over initial steps. Combined with cosine decay, it stabilizes early training. Use warmup_ratio=0.1 of total steps.",
                2: "Learning rate warmup prevents instability in early training when gradients are noisy. Common in large batch training and transformer models.",
                1: "Adam optimizer adapts learning rates per parameter using first and second moment estimates. AdamW decouples weight decay from the gradient update.",
                0: "Early stopping monitors validation loss and halts training when it stops improving. Patience determines how many epochs to wait.",
            },
        },
        {
            "query": "knowledge distillation teacher student",
            "docs": {
                3: "Knowledge distillation trains a small student model using soft labels from a large teacher. Temperature scaling smooths the teacher's output distribution. Loss combines hard and soft targets.",
                2: "Model compression reduces large model size for deployment. Techniques include pruning, quantization, knowledge distillation, and architecture search.",
                1: "Ensemble methods combine multiple models for better predictions. Averaging, weighted voting, and stacking are common ensemble strategies.",
                0: "Hyperparameter tuning searches for optimal model configuration. Grid search, random search, and Bayesian optimization are common approaches.",
            },
        },
        {
            "query": "positional encoding transformer sequence",
            "docs": {
                3: "Sinusoidal positional encodings add position information to token embeddings using sine and cosine functions. Learned positional embeddings are an alternative. Without position, attention is order-invariant.",
                2: "Transformers lack recurrence so they need explicit position information. Positional encodings are added to input embeddings to preserve sequence order.",
                1: "Tokenization splits text into subword units. BPE, WordPiece, and SentencePiece handle out-of-vocabulary words by breaking them into known subwords.",
                0: "Convolutional Neural Networks apply learnable filters to detect spatial patterns. Pooling layers reduce spatial dimensions while preserving important features.",
            },
        },
        {
            "query": "mixed precision training fp16 bf16",
            "docs": {
                3: "Mixed precision uses fp16 for forward/backward and fp32 for weight updates. Loss scaling prevents underflow. bf16 has larger dynamic range than fp16, reducing overflow risk.",
                2: "Mixed precision training reduces memory usage and speeds up computation on modern GPUs. Automatic mixed precision (AMP) manages dtype conversion.",
                1: "GPU memory management involves batch size, gradient accumulation, and activation checkpointing. OOM errors occur when memory exceeds GPU capacity.",
                0: "Distributed training splits work across multiple GPUs. Data parallelism replicates the model, while pipeline parallelism splits layers across devices.",
            },
        },
    ],
    "data_infrastructure": [
        {
            "query": "vector database indexing strategies",
            "docs": {
                3: "HNSW graphs provide fast approximate nearest neighbor search. IVF partitions vectors into clusters. PQ compresses vectors for memory efficiency at accuracy cost.",
                2: "Vector databases store and search high-dimensional embeddings. Index types include HNSW, IVF, IVF-PQ, and flat brute-force for different use cases.",
                1: "Embedding models convert text or images into dense vectors. Similar items have nearby vectors, enabling similarity search through cosine or Euclidean distance.",
                0: "Bloom filters probabilistically test set membership with false positives but no false negatives. Useful for pre-filtering before expensive lookups.",
            },
        },
        {
            "query": "data pipeline idempotency",
            "docs": {
                3: "Idempotent pipelines produce the same result when run multiple times. Use upsert operations, deterministic partitioning, and watermark-based deduplication.",
                2: "Pipeline reliability requires handling failures gracefully. Idempotency ensures reprocessing doesn't create duplicates through exactly-once semantics.",
                1: "Apache Spark processes large datasets using distributed computing. RDDs and DataFrames provide abstractions for parallel transformations.",
                0: "Data lakes store raw data in its native format until needed. Lakehouse architectures combine lake flexibility with warehouse query optimization.",
            },
        },
        {
            "query": "columnar storage parquet optimization",
            "docs": {
                3: "Parquet benefits from proper row group sizing (128MB-1GB), column pruning, predicate pushdown, and dictionary encoding. Use snappy or zstd compression.",
                2: "Apache Parquet is a columnar storage format optimized for analytical queries. It stores data by columns enabling efficient field-specific reads.",
                1: "CSV files store tabular data as plain text with delimiters. While human-readable, they lack schema enforcement and compression.",
                0: "ORC files are another columnar format optimized for Hive. They include built-in indexes and support ACID transactions.",
            },
        },
        {
            "query": "stream processing exactly-once semantics",
            "docs": {
                3: "Exactly-once processing requires idempotent writes and transactional reads. Kafka transactions with read_committed isolation and checkpointing achieve this guarantee.",
                2: "Stream processing semantics range from at-least-once (duplicates possible) to exactly-once (no duplicates). Exactly-once requires coordination between source and sink.",
                1: "Event time vs processing time: event time is when events occurred, processing time is when they're observed. Watermarks handle late-arriving data.",
                0: "Data warehousing schemas include star schema (fact and dimension tables) and snowflake schema (normalized dimensions). Choice affects query performance.",
            },
        },
        {
            "query": "database sharding partitioning strategy",
            "docs": {
                3: "Sharding splits data across machines using hash, range, or directory-based strategies. Handle hot keys with sub-sharding. Cross-shard queries require scatter-gather or denormalization.",
                2: "Horizontal partitioning divides tables by rows across storage units. Sharding is distributed partitioning across multiple database servers.",
                1: "Database replication copies data from primary to replicas for read scaling and failover. Synchronous replication ensures consistency but adds latency.",
                0: "CAP theorem states distributed systems can guarantee at most two of consistency, availability, and partition tolerance. Design choices depend on use case.",
            },
        },
        {
            "query": "data catalog metadata management",
            "docs": {
                3: "Data catalogs provide centralized metadata: schema, lineage, ownership, and quality metrics. Tools like DataHub and Amundsen enable data discovery and governance.",
                2: "Metadata management tracks data assets across an organization. Technical metadata includes schemas and formats. Business metadata includes definitions and stewards.",
                1: "Data quality dimensions include accuracy, completeness, consistency, timeliness, and validity. Automated checks catch issues before downstream impact.",
                0: "Feature stores manage ML features with offline/online consistency. They provide feature versioning, point-in-time correctness, and low-latency serving.",
            },
        },
        {
            "query": "CDC change data capture patterns",
            "docs": {
                3: "CDC captures row-level changes from database transaction logs (WAL, binlog). Debezium reads from MySQL, PostgreSQL, MongoDB logs. Enables real-time replication and event sourcing.",
                2: "Change Data Capture tracks inserts, updates, and deletes in databases. Log-based CDC reads transaction logs without impacting source performance.",
                1: "ETL pipelines extract, transform, and load data between systems. ELT reverses the order, loading raw data first then transforming in the warehouse.",
                0: "Message brokers decouple producers and consumers of events. Kafka provides durable, ordered, partitioned event streams with consumer group management.",
            },
        },
    ],
    "devops_infrastructure": [
        {
            "query": "kubernetes pod resource limits",
            "docs": {
                3: "Set both requests and limits for CPU and memory. Requests affect scheduling; limits enforce throttling or OOMKill. Set requests=limits for Guaranteed QoS class.",
                2: "Kubernetes resource management uses requests and limits to control scheduling and consumption. QoS class determines eviction priority during pressure.",
                1: "Kubernetes orchestrates containerized workloads across clusters. Pods are the smallest deployable units with shared network and storage.",
                0: "Docker multi-stage builds reduce image size by copying only needed artifacts from build stages. Each stage can use a different base image.",
            },
        },
        {
            "query": "CI/CD pipeline caching strategies",
            "docs": {
                3: "Cache dependencies using hash keys from lock files. Cache build outputs separately. Use partial cache restore for warm starts. Set TTL to avoid stale caches.",
                2: "CI/CD caching speeds up pipelines by storing reusable artifacts. Common types include dependency caches, build caches, and Docker layer caches.",
                1: "Continuous Integration automatically builds and tests code on every commit. Continuous Deployment releases passing builds to production.",
                0: "Git branching strategies include GitFlow, GitHub Flow, and trunk-based development. Each balances stability with delivery speed differently.",
            },
        },
        {
            "query": "infrastructure as code state management",
            "docs": {
                3: "Terraform state tracks config-to-resource mapping. Store remotely with locking. Use workspaces for isolation. Never commit state. Run plan before apply.",
                2: "Infrastructure as Code tools manage cloud resources declaratively. State files record existing resources for drift detection and planning.",
                1: "Cloud providers offer managed services for compute, storage, and networking. Infrastructure can be provisioned via console, CLI, or API.",
                0: "Monitoring systems collect metrics, logs, and traces. Prometheus pulls metrics, while agents push logs to centralized systems like ELK.",
            },
        },
        {
            "query": "docker container security scanning",
            "docs": {
                3: "Scan images for CVEs using Trivy, Grype, or Snyk. Pin base image digests, not tags. Use distroless or scratch images to minimize attack surface. Run as non-root user.",
                2: "Container security involves image scanning, runtime protection, and supply chain integrity. SBOMs document all components in an image.",
                1: "Container registries store and distribute Docker images. Private registries support access control and vulnerability scanning.",
                0: "Service meshes like Istio manage service-to-service communication with mTLS, traffic splitting, and observability.",
            },
        },
        {
            "query": "kubernetes ingress controller TLS termination",
            "docs": {
                3: "Ingress controllers route external traffic to services. Configure TLS with cert-manager for automatic certificate renewal. Use IngressClass to select the controller.",
                2: "Kubernetes networking uses Services, Ingress, and NetworkPolicies. Ingress provides HTTP/HTTPS routing rules for external access.",
                1: "Kubernetes ConfigMaps and Secrets store configuration and sensitive data. Secrets are base64-encoded, not encrypted by default.",
                0: "Load balancers distribute traffic across backend instances. Layer 4 (TCP/UDP) vs Layer 7 (HTTP) load balancing serve different needs.",
            },
        },
        {
            "query": "gitops argocd deployment strategy",
            "docs": {
                3: "GitOps uses Git as the source of truth for infrastructure. ArgoCD syncs cluster state to Git repos. Use application sets for multi-environment management.",
                2: "GitOps automates deployment through Git pull requests. The desired state is in Git, and operators reconcile the actual state to match.",
                1: "Blue-green deployments run two identical environments. Traffic switches from blue to green after validation, enabling instant rollback.",
                0: "Container orchestration platforms include Kubernetes, Docker Swarm, and Nomad. Kubernetes dominates with the largest ecosystem.",
            },
        },
        {
            "query": "observability SLO error budget monitoring",
            "docs": {
                3: "SLOs define reliability targets. Error budgets = 1 - SLO. When budget is exhausted, halt feature releases. Use SLIs like availability, latency, and correctness to measure.",
                2: "Observability combines metrics, logs, and traces. SLOs and error budgets translate reliability into actionable engineering decisions.",
                1: "Alerting should be actionable and page humans only when immediate action is needed. Use alert fatigue prevention with proper routing.",
                0: "Distributed tracing follows requests across services. OpenTelemetry provides vendor-neutral instrumentation for traces, metrics, and logs.",
            },
        },
    ],
    "security": [
        {
            "query": "JWT token security best practices",
            "docs": {
                3: "Store JWTs in httpOnly cookies, not localStorage. Use short expiration with refresh tokens. Validate signatures server-side with RS256. Never store sensitive data in payload.",
                2: "JSON Web Tokens consist of header, payload, and signature. They're used for authentication but must avoid vulnerabilities like token theft or algorithm confusion.",
                1: "OAuth 2.0 enables applications to obtain limited access to user accounts. It delegates authentication using grant types like authorization code.",
                0: "CORS controls which origins can access resources via browser. It's a browser security feature, not a server-side access control mechanism.",
            },
        },
        {
            "query": "SQL injection prevention methods",
            "docs": {
                3: "Use parameterized queries to prevent SQL injection. Never concatenate user input. Use ORMs that parameterize by default. Apply input validation as defense-in-depth.",
                2: "SQL injection occurs when untrusted input is included in queries without sanitization. Attackers can read, modify, or delete data from the database.",
                1: "Input validation checks that data meets expected format, type, and range. It should be performed on both client and server side.",
                0: "Stored procedures can encapsulate SQL logic but don't inherently prevent injection if they use dynamic SQL with string concatenation.",
            },
        },
        {
            "query": "zero trust network architecture",
            "docs": {
                3: "Zero trust assumes no implicit trust. Verify every request, enforce least privilege, micro-segment networks, and continuously validate identity and device health.",
                2: "Zero trust architecture replaces perimeter-based security with identity-centric controls. Every access request is authenticated and authorized regardless of source.",
                1: "Network segmentation divides networks into isolated zones. VLANs and subnets limit lateral movement if one segment is compromised.",
                0: "Firewalls filter network traffic based on rules. Stateful firewalls track connection state. Next-gen firewalls inspect application-layer content.",
            },
        },
        {
            "query": "secret management vault rotation",
            "docs": {
                3: "Store secrets in HashiCorp Vault or AWS Secrets Manager. Rotate credentials automatically. Never hardcode secrets. Use short-lived credentials and dynamic secrets where possible.",
                2: "Secret management involves secure storage, access control, and rotation of credentials, API keys, and certificates. Centralized vaults prevent secret sprawl.",
                1: "Environment variables pass configuration to applications. They're commonly used for secrets but lack encryption and access auditing.",
                0: "Password hashing algorithms like bcrypt, scrypt, and argon2 protect stored credentials. Salting prevents rainbow table attacks.",
            },
        },
        {
            "query": "CSP content security policy headers",
            "docs": {
                3: "CSP restricts resource loading sources. Use 'self' for scripts, specify allowed CDNs, and report violations with report-uri. Start with report-only mode to avoid breaking functionality.",
                2: "Content Security Policy headers prevent XSS by whitelisting allowed script, style, and image sources. Nonces and hashes enable inline content safely.",
                1: "HTTP security headers include HSTS, X-Frame-Options, X-Content-Type-Options, and Referrer-Policy. Each addresses different attack vectors.",
                0: "Rate limiting protects APIs from abuse by capping request frequency. Token bucket and sliding window algorithms implement rate limits.",
            },
        },
        {
            "query": "supply chain attack dependency verification",
            "docs": {
                3: "Pin dependency versions with lock files. Verify package checksums. Use SBOMs to track components. Monitor for known vulnerabilities. Avoid typosquatting with allowlists.",
                2: "Software supply chain attacks compromise dependencies to infect downstream applications. Risks include malicious packages, compromised maintainers, and CI/CD pipeline attacks.",
                1: "Dependency management tools like npm, pip, and Maven resolve and install packages. Transitive dependencies introduce indirect vulnerabilities.",
                0: "Code signing verifies software authenticity and integrity. Digital signatures ensure code hasn't been tampered with since signing.",
            },
        },
    ],
    "real_estate_singapore": [
        {
            "query": "singapore hdb eligibility first-timer",
            "docs": {
                3: "First-timers must be Singapore citizens aged 21+, form a family nucleus, and meet the $14,000/month income ceiling. They get priority balloting and CPF grants up to $80,000.",
                2: "HDB flats house about 80% of Singapore residents. Eligibility depends on citizenship, age, income ceiling, and family scheme. Various grants are available.",
                1: "Singapore's property market includes HDB flats, private condos, and landed properties. ABSD applies to foreigners purchasing residential property.",
                0: "Commercial properties in Singapore include offices, retail spaces, and industrial units. They have different financing rules from residential.",
            },
        },
        {
            "query": "singapore property loan LTV ratio",
            "docs": {
                3: "LTV limits: HDB loans up to 75% (CPF allowed), bank loans up to 75% (5% cash minimum). ABSD: 0% for SC first home, 20% second, 60% foreigners.",
                2: "Singapore mortgage regulations set maximum LTV ratios. TDSR caps monthly debt repayments at 55% of gross monthly income.",
                1: "Interest rates for Singapore home loans can be fixed or floating. Fixed rates lock for 2-3 years. Floating rates track SORA.",
                0: "Property decoupling transfers ownership between spouses to avoid ABSD on second property purchases. It requires careful tax and legal planning.",
            },
        },
        {
            "query": "singapore condo facilities maintenance fees",
            "docs": {
                3: "Condo maintenance fees range from $0.30 to $0.60 per sqft per month, covering pool, gym, security, and upkeep. Fees are set by the MCST.",
                2: "Private condominiums offer shared facilities like pools and gyms. Monthly maintenance fees fund upkeep and vary by development age and quality.",
                1: "Condominiums are subject to the Land Titles (Strata) Act. The MCST manages common property through annual general meetings.",
                0: "Housing prices in Singapore are influenced by location, MRT proximity, school catchment areas, and remaining lease length.",
            },
        },
        {
            "query": "singapore resale flat valuation process",
            "docs": {
                3: "HDB resale flats require a valuation report before purchase. Compare recent transacted prices on HDB portal. Factor in lease decay, renovation costs, and proximity to amenities.",
                2: "HDB resale market prices are determined by supply and demand. Location, floor level, remaining lease, and flat condition affect valuation.",
                1: "HDB grants like the Family Grant and Proximity Housing Grant help eligible buyers afford resale flats. Amounts vary by household type.",
                0: "Property agents in Singapore must be CEA-registered. They charge commission on resale transactions, typically 1-2% of the sale price.",
            },
        },
        {
            "query": "singapore BTO vs resale flat comparison",
            "docs": {
                3: "BTO flats are cheaper with longer leases but have 3-4 year wait times. Resale flats are move-in ready with mature estates but cost more. Consider grants, location, and timeline.",
                2: "BTO (Build-To-Order) flats are new HDB flats sold directly by the government. Resale flats are previously owned HDB flats sold on the open market.",
                1: "HDB MOP (Minimum Occupation Period) requires owners to live in their flat for 5 years before selling or buying private property.",
                0: "En bloc sales involve collective sale of all units in a development. Owners share proceeds based on strata share values.",
            },
        },
        {
            "query": "singapore property tax rates owner occupier",
            "docs": {
                3: "Owner-occupied property tax rates are progressive: 0% on first $8,000 AV, up to 32% on amounts above $130,000. Non-owner-occupied rates are higher across all brackets.",
                2: "Singapore property tax is based on Annual Value (AV), the estimated gross annual rent. Owner-occupiers enjoy lower tax rates than investors.",
                1: "Stamp duty applies to property transactions. BSD is based on purchase price. ABSD is an additional tax for certain buyer profiles.",
                0: "Rental agreements in Singapore typically require a 2-year lease with a 1-month security deposit. Tenants pay stamp duty on the lease.",
            },
        },
        {
            "query": "singapore EC executive condominium rules",
            "docs": {
                3: "ECs are hybrid public-private housing with a 5-year MOP. After 10 years they become fully privatized. Income ceiling is $16,000/month. EC grants are available for eligible buyers.",
                2: "Executive Condominiums combine HDB and private features. They're built by private developers but sold with HDB eligibility conditions.",
                1: "Private condominiums have no income ceiling or citizenship restrictions for purchase. Foreigners can buy condos but not HDB flats.",
                0: "Housing development in Singapore is planned by URA through the Master Plan and Concept Plan. Land use zoning determines what can be built.",
            },
        },
    ],
    "science_health": [
        {
            "query": "circadian rhythm sleep optimization",
            "docs": {
                3: "Optimize circadian rhythm with morning sunlight, avoid blue light 2 hours before bed, maintain consistent sleep times, and keep bedroom at 18-20°C.",
                2: "The circadian rhythm is a 24-hour internal clock regulating sleep-wake cycles. It's influenced by light exposure and disrupted by shift work or jet lag.",
                1: "Sleep consists of REM and non-REM cycles repeating every 90 minutes. Deep sleep is physically restorative. REM aids memory consolidation.",
                0: "Melatonin supplements can help with jet lag and delayed sleep phase. They're not a long-term solution for chronic insomnia.",
            },
        },
        {
            "query": "CRISPR gene editing mechanism",
            "docs": {
                3: "CRISPR-Cas9 uses guide RNA to direct Cas9 to specific DNA. It creates double-strand breaks repaired via NHEJ (knockouts) or HDR (precise edits).",
                2: "CRISPR is gene editing adapted from bacterial immune systems. It allows precise DNA modification with applications in medicine and agriculture.",
                1: "DNA contains genetic instructions for organism development. It consists of four bases (A, T, G, C) in a double helix with complementary pairing.",
                0: "Epigenetics studies heritable changes in gene expression without altering DNA sequence. Methylation and histone modification are key mechanisms.",
            },
        },
        {
            "query": "gut microbiome diet impact",
            "docs": {
                3: "Diet shapes gut microbiome composition within days. Fiber-rich diets promote diversity. Fermented foods introduce beneficial bacteria. Artificial sweeteners may disrupt microbial balance.",
                2: "The gut microbiome contains trillions of microorganisms affecting digestion, immunity, and mood. Diet is the primary modifiable factor.",
                1: "Probiotics contain live beneficial bacteria. Prebiotics are fibers that feed existing gut bacteria. Synbiotics combine both approaches.",
                0: "Antibiotics kill or inhibit bacterial growth. Broad-spectrum antibiotics affect both harmful and beneficial bacteria, disrupting the microbiome.",
            },
        },
        {
            "query": "intermittent fasting metabolic effects",
            "docs": {
                3: "Intermittent fasting triggers autophagy, improves insulin sensitivity, and shifts metabolism to fat oxidation. Common patterns include 16:8 and 5:2. Effects vary by individual.",
                2: "Fasting periods allow insulin levels to drop, promoting fat burning. Time-restricted eating aligns food intake with circadian rhythms.",
                1: "Caloric restriction extends lifespan in model organisms. Mechanisms include reduced mTOR signaling and increased sirtuin activity.",
                0: "Ketogenic diets are high-fat, very low-carb diets that induce ketosis. They differ from fasting in macronutrient composition.",
            },
        },
        {
            "query": "vaccine mRNA mechanism development",
            "docs": {
                3: "mRNA vaccines deliver genetic instructions for cells to produce viral proteins, triggering immune responses. Lipid nanoparticles protect mRNA and enable cellular uptake.",
                2: "mRNA technology encodes antigens that cells produce temporarily. The immune system learns to recognize the pathogen without exposure to the actual virus.",
                1: "Traditional vaccines use weakened or inactivated pathogens, or subunit proteins. They've eradicated or controlled many infectious diseases.",
                0: "Herd immunity occurs when enough of a population is immune to interrupt disease transmission. Thresholds vary by disease contagiousness.",
            },
        },
        {
            "query": "exercise cardiovascular benefits mechanisms",
            "docs": {
                3: "Aerobic exercise strengthens heart muscle, lowers resting heart rate, improves endothelial function, and increases HDL cholesterol. 150 min/week moderate activity is recommended.",
                2: "Regular exercise reduces cardiovascular disease risk through multiple pathways: blood pressure reduction, improved lipid profiles, and anti-inflammatory effects.",
                1: "VO2 max measures maximum oxygen consumption during exercise. It's a strong predictor of cardiovascular fitness and overall health.",
                0: "Resistance training builds muscle mass and bone density. It also improves insulin sensitivity and resting metabolic rate.",
            },
        },
    ],
    "finance": [
        {
            "query": "index fund vs ETF investing",
            "docs": {
                3: "Index funds and ETFs both track indices but differ in trading: ETFs trade intraday with spreads, index funds price at NAV. ETFs have lower fees and are more tax-efficient.",
                2: "Passive investing through index funds and ETFs provides broad market exposure at low cost. Both track benchmarks like the S&P 500.",
                1: "Dollar-cost averaging invests fixed amounts at regular intervals regardless of market conditions. This reduces volatility impact.",
                0: "Mutual funds are actively managed portfolios pooling investor money. They charge higher expense ratios than passive index funds.",
            },
        },
        {
            "query": "compound interest investment growth",
            "docs": {
                3: "Compound interest: A = P(1 + r/n)^(nt). Rule of 72: 72/rate estimates doubling time. At 7%, money doubles every ~10 years. Starting early matters most.",
                2: "Compound interest earns returns on principal and accumulated returns. Over long periods, exponential growth outpaces simple interest significantly.",
                1: "Inflation erodes purchasing power over time. Real returns = nominal - inflation. Historical stock returns average ~7% after inflation.",
                0: "Annuities provide guaranteed income streams in retirement. Fixed annuities offer stable payments, variable annuities depend on investment performance.",
            },
        },
        {
            "query": "portfolio diversification correlation risk",
            "docs": {
                3: "Diversification reduces risk by combining uncorrelated assets. Stocks, bonds, real estate, and commodities have different return drivers. Correlation increases during crises.",
                2: "Modern Portfolio Theory optimizes risk-return trade-offs through asset allocation. The efficient frontier represents optimal portfolios.",
                1: "Risk tolerance determines appropriate asset allocation. Younger investors can typically take more equity risk due to longer time horizons.",
                0: "Market timing attempts to buy low and sell high by predicting market movements. Academic evidence shows it underperforms buy-and-hold.",
            },
        },
        {
            "query": "tax loss harvesting investment strategy",
            "docs": {
                3: "Tax loss harvesting sells losing positions to offset capital gains. Watch for wash sale rules (30-day window). Harvest losses to reduce taxable income up to $3,000/year.",
                2: "Tax-efficient investing minimizes tax drag on returns. Strategies include holding in tax-advantaged accounts, tax loss harvesting, and favoring qualified dividends.",
                1: "Capital gains tax applies when selling appreciated assets. Short-term gains (under 1 year) are taxed as ordinary income. Long-term rates are lower.",
                0: "Roth IRA contributions are made with after-tax dollars but grow tax-free. Traditional IRA contributions may be tax-deductible.",
            },
        },
        {
            "query": "bond duration interest rate sensitivity",
            "docs": {
                3: "Duration measures bond price sensitivity to rate changes. A 5-year duration means ~5% price drop for 1% rate rise. Longer duration = more rate risk. Zero-coupon bonds have highest duration.",
                2: "Bond prices move inversely to interest rates. Duration and convexity quantify this relationship. Rising rates hurt existing bond values.",
                1: "Credit ratings from Moody's and S&P assess default risk. Investment grade (BBB-/Baa3+) vs high yield (junk) bonds have different risk profiles.",
                0: "Stock valuation methods include DCF, P/E multiples, and dividend discount models. Each has assumptions about growth and risk.",
            },
        },
        {
            "query": "emergency fund size liquidity planning",
            "docs": {
                3: "Emergency funds should cover 3-6 months of essential expenses. Keep in high-yield savings or money market accounts. Self-employed or variable income earners should target 6-12 months.",
                2: "Emergency funds provide financial security against unexpected expenses or income loss. They prevent high-interest debt during crises.",
                1: "High-yield savings accounts offer better rates than traditional savings. Money market accounts provide similar returns with check-writing.",
                0: "Personal budgeting methods include 50/30/20 rule, zero-based budgeting, and envelope system. Tracking spending is the first step.",
            },
        },
        {
            "query": "real estate REIT dividend investing",
            "docs": {
                3: "REITs must distribute 90% of taxable income as dividends, yielding 4-8%. They offer real estate exposure without property management. Equity REITs own properties, mortgage REITs lend.",
                2: "Real Estate Investment Trusts pool capital to own income-producing properties. They trade like stocks and provide liquidity unavailable in direct real estate.",
                1: "Dividend investing focuses on stocks with consistent dividend payments. Dividend aristocrats have increased payouts for 25+ consecutive years.",
                0: "Options strategies like covered calls generate income from stock holdings. Selling puts can acquire stocks at lower prices.",
            },
        },
    ],
    "legal": [
        {
            "query": "contract breach remedies damages",
            "docs": {
                3: "Contract breach remedies include compensatory damages (actual losses), consequential damages (foreseeable losses), and specific performance (court-ordered fulfillment). Liquidated damages must be reasonable estimates.",
                2: "When a contract is breached, the non-breaching party can seek damages to be made whole. The goal is to put them in the position they would have been in had the contract been performed.",
                1: "Contracts require offer, acceptance, consideration, and mutual assent. Written contracts are easier to enforce than oral agreements.",
                0: "Tort law covers civil wrongs like negligence, defamation, and trespass. Unlike contracts, torts don't require a prior agreement between parties.",
            },
        },
        {
            "query": "intellectual property patent copyright trademark",
            "docs": {
                3: "Patents protect inventions (20 years), copyrights protect creative works (life + 70 years), trademarks protect brand identifiers (renewable). Trade secrets protect confidential business information indefinitely.",
                2: "Intellectual property law protects creations of the mind. Different IP types cover different kinds of creations with different durations and requirements.",
                1: "Fair use allows limited use of copyrighted material without permission for commentary, criticism, education, and research. Four factors determine fair use.",
                0: "Employment law governs the relationship between employers and employees, including wages, discrimination, and workplace safety regulations.",
            },
        },
        {
            "query": "non-disclosure agreement NDA enforceability",
            "docs": {
                3: "NDAs must define confidential information clearly, specify duration, and include exclusions (public knowledge, independently developed). Overly broad NDAs may be unenforceable. Mutual NDAs protect both parties.",
                2: "Non-disclosure agreements legally bind parties to keep shared information confidential. They're common in business negotiations, employment, and partnerships.",
                1: "Non-compete agreements restrict employees from working for competitors. Many jurisdictions are limiting or banning non-competes to protect worker mobility.",
                0: "Arbitration clauses require disputes to be resolved through private arbitration rather than court. They're common in consumer and employment contracts.",
            },
        },
        {
            "query": "data privacy GDPR compliance requirements",
            "docs": {
                3: "GDPR requires lawful basis for processing, data subject rights (access, erasure, portability), breach notification within 72 hours, and DPO appointment for large-scale processing. Fines up to 4% of global revenue.",
                2: "The General Data Protection Regulation governs personal data processing of EU residents. It applies to any organization processing EU resident data regardless of location.",
                1: "Data minimization means collecting only data necessary for the stated purpose. Purpose limitation prevents using data for unrelated purposes.",
                0: "Cybersecurity frameworks like NIST and ISO 27001 provide guidelines for protecting information systems. They're not legally mandated but are industry best practices.",
            },
        },
        {
            "query": "corporate liability limited company LLC",
            "docs": {
                3: "LLCs provide limited liability protection separating personal and business assets. Members aren't personally liable for business debts. Piercing the corporate veil occurs when formalities aren't maintained.",
                2: "Business structures include sole proprietorships, partnerships, LLCs, and corporations. Each has different liability, tax, and governance implications.",
                1: "Corporate governance involves board oversight, shareholder rights, and fiduciary duties. Directors owe duties of care and loyalty to the corporation.",
                0: "Bankruptcy law provides relief for debtors unable to pay creditors. Chapter 7 liquidates assets, Chapter 11 reorganizes, Chapter 13 creates repayment plans.",
            },
        },
        {
            "query": "employment at-will termination rights",
            "docs": {
                3: "At-will employment means either party can terminate the relationship at any time for any legal reason. Exceptions include discrimination, retaliation, public policy violations, and implied contracts.",
                2: "Employment termination can be voluntary (resignation) or involuntary (layoff, firing). Wrongful termination claims arise when firing violates law or contract.",
                1: "Severance packages provide compensation upon termination. They're not legally required but are often offered in exchange for release of claims.",
                0: "Workers' compensation provides benefits for work-related injuries regardless of fault. It's a no-fault system that limits employee lawsuits against employers.",
            },
        },
        {
            "query": "statute of limitations civil claims",
            "docs": {
                3: "Statutes of limitations set deadlines for filing lawsuits. They vary by claim type and jurisdiction: personal injury (1-3 years), contracts (3-6 years), property damage (2-4 years). Tolling can pause the clock.",
                2: "The statute of limitations prevents stale claims by requiring legal action within a specified time. The clock typically starts when the harm occurs or is discovered.",
                1: "Jurisdiction determines which court can hear a case. Personal jurisdiction requires minimum contacts with the forum state. Subject matter jurisdiction depends on the type of claim.",
                0: "Class action lawsuits allow groups with similar claims to sue collectively. They're efficient for small individual damages that would be impractical to litigate separately.",
            },
        },
    ],
    "education": [
        {
            "query": "spaced repetition memory retention",
            "docs": {
                3: "Spaced repetition schedules reviews at increasing intervals based on the forgetting curve. Tools like Anki use algorithms to optimize review timing. It's more effective than massed practice (cramming).",
                2: "The spacing effect shows that information reviewed over spaced intervals is retained longer than information studied in a single session.",
                1: "Active recall involves retrieving information from memory rather than re-reading. Practice tests and flashcards are forms of active recall.",
                0: "Learning styles theory (visual, auditory, kinesthetic) lacks empirical support. Evidence shows matching instruction to preferred style doesn't improve outcomes.",
            },
        },
        {
            "query": "formative vs summative assessment",
            "docs": {
                3: "Formative assessments occur during learning to guide instruction (quizzes, discussions, exit tickets). Summative assessments evaluate learning at the end (exams, final projects). Both serve different purposes.",
                2: "Assessment types include diagnostic (before learning), formative (during), and summative (after). Effective teaching uses all three to support student learning.",
                1: "Rubrics define performance criteria and levels for assignments. Analytic rubrics score multiple dimensions. Holistic rubrics provide an overall score.",
                0: "Standardized tests measure student performance against norms or standards. They're used for accountability but criticized for narrowing curriculum.",
            },
        },
        {
            "query": "Bloom taxonomy cognitive levels",
            "docs": {
                3: "Bloom's revised taxonomy: remember, understand, apply, analyze, evaluate, create. Learning objectives should target higher-order thinking. Verbs like 'analyze' and 'evaluate' indicate deeper cognition.",
                2: "Bloom's Taxonomy classifies educational objectives by cognitive complexity. The revised version replaces nouns with verbs and reorders the top two levels.",
                1: "Learning objectives should be specific, measurable, and aligned with assessments. Well-written objectives guide both teaching and evaluation.",
                0: "Constructivism posits that learners actively construct knowledge through experience and reflection. It contrasts with behaviorist transmission models.",
            },
        },
        {
            "query": "universal design learning UDL principles",
            "docs": {
                3: "UDL provides multiple means of engagement, representation, and action/expression. It proactively designs for learner variability rather than retrofitting accommodations.",
                2: "Universal Design for Learning creates accessible curricula for all learners. It's based on neuroscience research showing diverse learning networks.",
                1: "Accommodations modify how students access or demonstrate learning (extra time, alternative formats). Modifications change what students are expected to learn.",
                0: "Differentiated instruction tailors teaching to individual student needs through content, process, product, and learning environment adjustments.",
            },
        },
        {
            "query": "growth mindset vs fixed mindset education",
            "docs": {
                3: "Growth mindset believes abilities develop through effort and strategy. Fixed mindset sees ability as innate. Praise effort and process, not intelligence. Mindset interventions show modest but meaningful effects.",
                2: "Carol Dweck's mindset theory distinguishes beliefs about ability. Growth mindset students embrace challenges and view failure as learning opportunities.",
                1: "Self-efficacy is belief in one's capability to succeed in specific situations. It's task-specific and influenced by mastery experiences and social modeling.",
                0: "Intrinsic motivation drives behavior from internal satisfaction. Extrinsic motivation relies on external rewards. Overjustification effect can reduce intrinsic motivation.",
            },
        },
        {
            "query": "project-based learning implementation",
            "docs": {
                3: "PBL engages students in extended inquiry around complex questions. Key elements: driving question, sustained inquiry, authenticity, student voice/choice, reflection, and public product.",
                2: "Project-based learning has students work on real-world problems over extended periods. It develops critical thinking, collaboration, and communication skills.",
                1: "Inquiry-based learning starts with questions rather than facts. Students investigate, research, and construct understanding through exploration.",
                0: "Flipped classrooms deliver instruction at home (videos) and practice in class. It maximizes teacher-student interaction during problem-solving.",
            },
        },
        {
            "query": "scaffolding zone proximal development",
            "docs": {
                3: "Vygotsky's ZPD is the gap between independent and guided performance. Scaffolding provides temporary support within the ZPD, gradually removed as competence increases. Examples: modeling, prompts, worked examples.",
                2: "Instructional scaffolding supports learners through complex tasks. Effective scaffolding is contingent (matched to need), fades over time, and transfers responsibility to the learner.",
                1: "Peer tutoring pairs students of different ability levels. Both tutor and tutee benefit academically. Cross-age and same-age tutoring are common formats.",
                0: "Metacognition involves thinking about one's own thinking. Metacognitive strategies include planning, monitoring, and evaluating one's learning process.",
            },
        },
    ],
}
