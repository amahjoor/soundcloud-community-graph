import os
import argparse
from scraper import SoundCloudScraper
from visualizer import SoundCloudVisualizer

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='SoundCloud Community Mapping Pipeline')
    
    # Common arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory to store/load data')
    
    # Subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')
    
    # Scraper arguments
    scraper_parser = subparsers.add_parser('scrape', help='Run the scraper')
    scraper_parser.add_argument('--seed_accounts', type=str, nargs='+',
                              help='Initial SoundCloud accounts to start with')
    scraper_parser.add_argument('--max_accounts', type=int, default=100,
                              help='Maximum number of accounts to scrape')
    scraper_parser.add_argument('--max_items', type=int, default=500,
                              help='Maximum number of items to collect per interaction type')
    scraper_parser.add_argument('--resume', action='store_true',
                              help='Resume scraping from last position')
    scraper_parser.add_argument('--chrome_path', type=str,
                              help='Path to Chrome binary')
    scraper_parser.add_argument('--browser', type=str, choices=['safari', 'chrome'], default='safari',
                              help='Browser to use for scraping (default: safari)')
    
    # Visualizer arguments
    vis_parser = subparsers.add_parser('visualize', help='Create visualizations')
    vis_parser.add_argument('--dim_reduction', type=str, default='umap',
                          choices=['umap', 'tsne', 'svd'],
                          help='Dimensionality reduction method')
    vis_parser.add_argument('--clustering', type=str, default='hdbscan',
                          choices=['hdbscan', 'dbscan', 'kmeans'],
                          help='Clustering method')
    vis_parser.add_argument('--min_interactions', type=int, default=10,
                          help='Minimum interactions for an account to be included')
    vis_parser.add_argument('--max_dominance', type=float, default=0.4,
                          help='Maximum percentage of interactions that can go to a single account')
    vis_parser.add_argument('--no_interactive', action='store_true',
                          help='Skip creating interactive plot')
    vis_parser.add_argument('--no_clusters', action='store_true',
                          help='Skip showing clusters')
    
    # Pipeline arguments
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument('--seed_accounts', type=str, nargs='+',
                               help='Initial SoundCloud accounts to start with')
    pipeline_parser.add_argument('--max_accounts', type=int, default=100,
                               help='Maximum number of accounts to scrape')
    pipeline_parser.add_argument('--max_items', type=int, default=500,
                               help='Maximum number of items to collect per interaction type')
    pipeline_parser.add_argument('--dim_reduction', type=str, default='umap',
                               choices=['umap', 'tsne', 'svd'],
                               help='Dimensionality reduction method')
    pipeline_parser.add_argument('--clustering', type=str, default='hdbscan',
                               choices=['hdbscan', 'dbscan', 'kmeans'],
                               help='Clustering method')
    pipeline_parser.add_argument('--min_interactions', type=int, default=10,
                               help='Minimum interactions for an account to be included')
    pipeline_parser.add_argument('--max_dominance', type=float, default=0.4,
                               help='Maximum percentage of interactions that can go to a single account')
    pipeline_parser.add_argument('--no_interactive', action='store_true',
                               help='Skip creating interactive plot')
    pipeline_parser.add_argument('--no_clusters', action='store_true',
                               help='Skip showing clusters')
    pipeline_parser.add_argument('--chrome_path', type=str,
                               help='Path to Chrome binary')
    pipeline_parser.add_argument('--browser', type=str, choices=['safari', 'chrome'], default='safari',
                               help='Browser to use for scraping (default: safari)')
    
    return parser.parse_args()

def run_scraper(args):
    """Run the SoundCloud scraper."""
    print("\n====== Starting SoundCloud Scraper ======\n")
    
    # Initialize scraper
    scraper = SoundCloudScraper(
        data_dir=args.data_dir,
        max_items=args.max_items,
        chrome_path=args.chrome_path if hasattr(args, 'chrome_path') else None,
        browser=args.browser if hasattr(args, 'browser') else 'safari'
    )
    
    # Run scraper
    if args.resume:
        scraper.resume(max_accounts=args.max_accounts)
    else:
        if not args.seed_accounts:
            # Default seed accounts if none provided
            seed_accounts = ["diplo", "flume", "kaytranada", "fourtet", "charlesthefirst"]
            print(f"No seed accounts provided, using defaults: {', '.join(seed_accounts)}")
        else:
            seed_accounts = args.seed_accounts
            
        scraper.scrape(
            seed_accounts=seed_accounts,
            max_accounts=args.max_accounts
        )
    
    print("\n====== SoundCloud Scraper Complete ======\n")

def run_visualizer(args):
    """Run the SoundCloud visualizer."""
    print("\n====== Starting SoundCloud Visualizer ======\n")
    
    # Initialize visualizer
    visualizer = SoundCloudVisualizer(data_dir=args.data_dir)
    
    # Create visualizations
    visualizer.visualize(
        dim_reduction=args.dim_reduction,
        clustering=args.clustering,
        min_interactions=args.min_interactions,
        max_dominance=args.max_dominance,
        interactive=not args.no_interactive,
        show_clusters=not args.no_clusters
    )
    
    print("\n====== SoundCloud Visualizer Complete ======\n")

def run_pipeline(args):
    """Run the full SoundCloud data pipeline."""
    # First run the scraper
    run_scraper(args)
    
    # Then run the visualizer
    run_visualizer(args)

def main():
    """Main function to run the pipeline."""
    args = parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Run the requested mode
    if args.mode == 'scrape':
        run_scraper(args)
    elif args.mode == 'visualize':
        run_visualizer(args)
    elif args.mode == 'pipeline':
        run_pipeline(args)
    else:
        print("Please specify a mode: scrape, visualize, or pipeline")
        print("Run 'python main.py -h' for more information")

if __name__ == "__main__":
    main() 