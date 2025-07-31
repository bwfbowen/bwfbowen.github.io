// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "dropdown-blog",
              title: "blog",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/blog/";
              },
            },{id: "dropdown-publications",
              title: "publications",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/publications/";
              },
            },{id: "dropdown-cv",
              title: "cv",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/cv/";
              },
            },{id: "dropdown-projects",
              title: "projects",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/projects/";
              },
            },{id: "dropdown-repositories",
              title: "repositories",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/repositories/";
              },
            },{id: "dropdown-teaching",
              title: "teaching",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/teaching/";
              },
            },{id: "post-some-recent-advancement-around-muzero",
        
          title: "Some Recent Advancement Around MuZero",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/blog-more-mcts/";
          
        },
      },{id: "post-mpc-with-a-differentiable-forward-model-an-implementation-with-jax",
        
          title: "MPC with a Differentiable Forward Model: An Implementation with Jax",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/blog-mpc/";
          
        },
      },{id: "post-adding-muzero-into-rl-toolkits-at-ease",
        
          title: "Adding MuZero into RL Toolkits at Ease",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/blog-muax/";
          
        },
      },{id: "post-hindsight-an-easy-yet-effective-rl-technique-her-with-pytorch-implementation",
        
          title: "“Hindsight” – An easy yet effective RL Technique HER with Pytorch implementation",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/blog-hindsight/";
          
        },
      },{id: "post-what-are-the-effective-deep-learning-models-for-tabular-data",
        
          title: "What are the Effective Deep Learning Models for Tabular Data?",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/blog-tabulardl/";
          
        },
      },{id: "post-will-drl-make-profit-in-high-frequency-trading",
        
          title: "Will DRL Make Profit in High-Frequency Trading?",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2021/blog-drlhft/";
          
        },
      },{id: "news-our-paper-slamuzero-plan-and-learn-to-map-for-joint-slam-and-navigation-was-accepted-to-icaps-2024",
          title: 'Our paper, SLAMuZero: Plan and learn to Map for Joint SLAM and Navigation,...',
          description: "",
          section: "News",},{id: "news-i-am-excited-to-begin-my-ph-d-studies-at-columbia-university",
          title: 'I am excited to begin my Ph.D. studies at Columbia University.',
          description: "",
          section: "News",},{id: "news-our-paper-efficient-consistency-model-training-for-policy-distillation-in-reinforcement-learning-was-accepted-to-the-iclr-2025-delta-workshop-as-a-poster-presentation",
          title: 'Our paper, Efficient Consistency Model Training for Policy Distillation in Reinforcement Learning, was...',
          description: "",
          section: "News",},{id: "news-i-am-honored-to-be-a-winner-of-the-cs3-validate-accelerator-program-which-will-provide-continued-funding-for-our-work-sina",
          title: 'I am honored to be a winner of the CS3 VALIDATE Accelerator program,...',
          description: "",
          section: "News",},{id: "news-i-will-be-starting-my-applied-scientist-internship-at-aws-ai-lab-this-summer",
          title: 'I will be starting my Applied Scientist Internship at AWS AI Lab this...',
          description: "",
          section: "News",},{id: "projects-muax",
          title: 'Muax',
          description: "An open-source implementation of MuZero with JAX and TensorFlow.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/muax/";
            },},{id: "projects-sina",
          title: 'SINA',
          description: "Seamless, Intelligent Navigation Anywhere",
          section: "Projects",handler: () => {
              window.location.href = "/projects/sina/";
            },},{id: "teaching-ta-optimization-models-and-methods",
          title: 'TA: Optimization Models and Methods',
          description: "TA for IEOR E4004. Topics included linear, nonlinear, integer, and dynamic programming.",
          section: "Teaching",handler: () => {
              window.location.href = "/teaching/2023-spring-optimization";
            },},{id: "teaching-ta-big-data-in-transportation",
          title: 'TA: Big Data in Transportation',
          description: "TA for CIEN E4011. Topics included high-performance computing with JAX, Google Cloud Platform, machine learning fundamentals, and model interpretability.",
          section: "Teaching",handler: () => {
              window.location.href = "/teaching/2025-spring-big-data";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%62%66%32%35%30%34@%63%6F%6C%75%6D%62%69%61.%65%64%75", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/bwfbowen", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/bowenfang", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=bEebg80AAAAJ", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
