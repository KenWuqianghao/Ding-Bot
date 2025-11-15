"""Dataset download utilities."""
import os
import pandas as pd
from typing import Optional
import chess
from tqdm import tqdm


def download_lichess_dataset(output_dir: str, limit: Optional[int] = None) -> str:
    """
    Download and process Lichess dataset with Stockfish evaluations from Hugging Face.
    
    Dataset: https://huggingface.co/datasets/Lichess/chess-position-evaluations
    
    Args:
        output_dir: Directory to save processed data
        limit: Optional limit on number of positions to process
        
    Returns:
        Path to saved CSV file
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets library not installed.")
        print("Install with: pip install datasets")
        raise
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Lichess chess position evaluations dataset from Hugging Face...")
    print("This may take a while on first download (dataset is ~37GB)...")
    
    # Load dataset
    # Note: This loads the full dataset. For testing, use streaming=True and limit
    dataset = load_dataset("Lichess/chess-position-evaluations", split="train")
    
    print(f"Dataset loaded. Total rows: {len(dataset):,}")
    
    positions = []
    evaluations = []
    best_moves = []
    
    # Process dataset
    print("Processing positions...")
    
    # Determine how many to process
    num_to_process = limit if limit else len(dataset)
    num_to_process = min(num_to_process, len(dataset))
    
    # Iterate over dataset
    for i in tqdm(range(num_to_process), desc="Processing"):
        example = dataset[i]
        fen = example['fen']
        
        # Get evaluation (cp or mate)
        # cp is centipawns, mate is moves to mate (positive for white, negative for black)
        cp = example.get('cp')
        mate = example.get('mate')
        
        if mate is not None:
            # Convert mate to centipawns (large value)
            # Mate in N moves: use large centipawn value
            # Positive mate = white wins, negative = black wins
            evaluation = 10000 if mate > 0 else -10000
        elif cp is not None:
            evaluation = float(cp)
        else:
            # Skip if no evaluation
            continue
        
        # Extract best move from principal variation line
        line = example.get('line', '')
        best_move = None
        if line:
            # Line is space-separated UCI moves, first move is best move
            moves = line.strip().split()
            if moves:
                best_move = moves[0]
        
        positions.append(fen)
        evaluations.append(evaluation)
        best_moves.append(best_move)
    
    # Create DataFrame
    df = pd.DataFrame({
        'fen': positions,
        'evaluation': evaluations,
        'best_move': best_moves
    })
    
    # Save to CSV
    output_path = os.path.join(output_dir, "lichess_positions.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\nProcessed {len(df):,} positions")
    print(f"Saved to {output_path}")
    print(f"\nDataset statistics:")
    print(f"  - Positions with evaluations: {len(df):,}")
    print(f"  - Positions with best moves: {df['best_move'].notna().sum():,}")
    print(f"  - Evaluation range: {df['evaluation'].min():.0f} to {df['evaluation'].max():.0f} centipawns")
    
    return output_path


def download_lichess_dataset_streaming(
    output_dir: str,
    limit: Optional[int] = None,
    batch_size: int = 10000,
    save_batches: bool = True
) -> str:
    """
    Download Lichess dataset using streaming (memory-efficient for large datasets).
    Processes in batches and saves incrementally to handle the full dataset (~784M positions).
    
    Args:
        output_dir: Directory to save processed data
        limit: Optional limit on number of positions (None = download all)
        batch_size: Batch size for processing and saving
        save_batches: If True, save incrementally to avoid memory issues
        
    Returns:
        Path to saved CSV file
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets library not installed.")
        print("Install with: pip install datasets")
        raise
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "lichess_positions.csv")
    
    if limit:
        print(f"Loading Lichess dataset (streaming mode, limit: {limit:,})...")
    else:
        print("Loading ENTIRE Lichess dataset (streaming mode)...")
        print("⚠️  WARNING: This will download ~784M positions (~37GB)")
        print("   This may take several hours depending on your internet connection.")
        print("   Processing will be done in batches to manage memory.")
    
    # Load with streaming
    dataset = load_dataset(
        "Lichess/chess-position-evaluations",
        split="train",
        streaming=True
    )
    
    positions = []
    evaluations = []
    best_moves = []
    
    print(f"Processing positions (batch size: {batch_size:,})...")
    count = 0
    batch_num = 0
    total_processed = 0
    
    # Track if file exists (for appending)
    file_exists = os.path.exists(output_path)
    write_header = not file_exists
    
    try:
        for example in tqdm(dataset, desc="Processing", unit=" positions"):
            if limit and count >= limit:
                break
            
            fen = example['fen']
            
            # Get evaluation
            cp = example.get('cp')
            mate = example.get('mate')
            
            if mate is not None:
                evaluation = 10000 if mate > 0 else -10000
            elif cp is not None:
                evaluation = float(cp)
            else:
                continue  # Skip positions without evaluation
            
            # Extract best move
            line = example.get('line', '')
            best_move = None
            if line:
                moves = line.strip().split()
                if moves:
                    best_move = moves[0]
            
            positions.append(fen)
            evaluations.append(evaluation)
            best_moves.append(best_move)
            count += 1
            
            # Save in batches to manage memory
            if save_batches and count % batch_size == 0:
                batch_num += 1
                df_batch = pd.DataFrame({
                    'fen': positions,
                    'evaluation': evaluations,
                    'best_move': best_moves
                })
                
                # Append to CSV (or create if first batch)
                df_batch.to_csv(
                    output_path,
                    mode='a' if not write_header else 'w',
                    header=write_header,
                    index=False
                )
                write_header = False
                
                total_processed += len(df_batch)
                print(f"  Batch {batch_num}: Saved {len(df_batch):,} positions (Total: {total_processed:,})")
                
                # Clear batch
                positions = []
                evaluations = []
                best_moves = []
        
        # Save remaining positions
        if positions:
            batch_num += 1
            df_batch = pd.DataFrame({
                'fen': positions,
                'evaluation': evaluations,
                'best_move': best_moves
            })
            
            df_batch.to_csv(
                output_path,
                mode='a' if not write_header else 'w',
                header=write_header,
                index=False
            )
            total_processed += len(df_batch)
            print(f"  Final batch {batch_num}: Saved {len(df_batch):,} positions")
        
        print(f"\n✓ Processing complete!")
        print(f"  Total positions processed: {total_processed:,}")
        print(f"  Saved to: {output_path}")
        
        # Get file size
        file_size_gb = os.path.getsize(output_path) / (1024**3)
        print(f"  File size: {file_size_gb:.2f} GB")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Download interrupted by user")
        if positions:
            print(f"  Saving {len(positions):,} remaining positions...")
            df_batch = pd.DataFrame({
                'fen': positions,
                'evaluation': evaluations,
                'best_move': best_moves
            })
            df_batch.to_csv(
                output_path,
                mode='a' if not write_header else 'w',
                header=write_header,
                index=False
            )
            total_processed += len(df_batch)
        print(f"  Total positions saved: {total_processed:,}")
        print(f"  You can resume by running the script again (it will append)")
    
    return output_path


def process_pgn_file(pgn_path: str, output_path: str, positions_per_game: int = 5, limit: Optional[int] = None):
    """
    Process PGN file to extract positions with evaluations.
    
    This function would use Stockfish to evaluate positions from games.
    Requires Stockfish to be installed and accessible.
    
    Args:
        pgn_path: Path to PGN file
        output_path: Path to save CSV
        positions_per_game: Number of positions to sample per game
        limit: Optional limit on total positions
    """
    import random
    from typing import List
    
    positions: List[str] = []
    evaluations: List[float] = []
    best_moves: List[str] = []
    
    print(f"Processing PGN file: {pgn_path}")
    print("Note: This requires Stockfish integration for evaluations")
    print("For now, creating placeholder structure...")
    
    # Placeholder implementation
    # In practice, you would:
    # 1. Parse PGN file
    # 2. For each game, sample positions
    # 3. Use Stockfish to evaluate positions and get best moves
    # 4. Save to CSV
    
    df = pd.DataFrame({
        'fen': positions,
        'evaluation': evaluations,
        'best_move': best_moves
    })
    
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
