import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SCPCreator:
    def __init__(self, dataset_name, use_relative_paths=True):
        """
        Initialize SCP creator with environment-based configuration
        
        Args:
            dataset_name: Name of the dataset (e.g., 'DNS', 'WSJ0', 'WHAM', etc.)
            use_relative_paths: If True, store relative paths in SCP files
        """
        self.dataset_name = dataset_name.upper()
        self.use_relative_paths = use_relative_paths
        self.scp_root = os.getenv('SCP_ROOT', 'data/scp')
        self.db_root = self._get_db_root()
        
    def _get_db_root(self):
        """Get database root path from environment variables"""
        env_key = f"{self.dataset_name}_DB_ROOT"
        db_root = os.getenv(env_key)
        if not db_root:
            raise ValueError(f"Database root not found for {self.dataset_name}. Please set {env_key} in .env file")
        return db_root
    
    def _get_dataset_path(self, subset_type):
        """Get specific dataset path from environment variables"""
        env_key = f"{self.dataset_name}_{subset_type.upper()}_PATH"
        path = os.getenv(env_key, "")
        return os.path.join(self.db_root, path) if path else self.db_root
    
    def _format_path(self, filepath):
        """Format path based on relative/absolute preference"""
        if self.use_relative_paths:
            # Convert to relative path from project root
            try:
                return os.path.relpath(filepath, os.getcwd())
            except ValueError:
                # If on different drives or cannot compute relative path
                return filepath
        return filepath
    
    def create_scp_file(self, subset_type, output_suffix, file_extension='.wav'):
        """
        Create SCP file for a specific subset of data
        
        Args:
            subset_type: Type of subset (e.g., 'clean', 'noise', 'train', 'dev', 'test')
            output_suffix: Suffix for output SCP filename (e.g., 'tr_s', 'cv_s', 'tr_n')
            file_extension: Extension of files to include (default: '.wav')
        """
        # Create output directory
        scp_dir = os.path.join(self.scp_root, f"scp_{self.dataset_name}")
        os.makedirs(scp_dir, exist_ok=True)
        
        # Get data path
        data_path = self._get_dataset_path(subset_type)
        if not os.path.exists(data_path):
            print(f"Warning: Path does not exist: {data_path}")
            return
        
        # Create SCP file
        scp_filepath = os.path.join(scp_dir, f"{output_suffix}.scp")
        with open(scp_filepath, 'w') as scp_file:
            for root, dirs, files in os.walk(data_path):
                files.sort()
                for file in files:
                    if file.endswith(file_extension):
                        full_path = os.path.join(root, file)
                        formatted_path = self._format_path(full_path)
                        # Write as: filename relative_or_absolute_path
                        scp_file.write(f"{file} {formatted_path}\n")
        
        print(f"Created SCP file: {scp_filepath}")
        return scp_filepath
    
    def create_standard_scp_files(self):
        """Create standard train/dev/test SCP files based on dataset type"""
        if self.dataset_name == 'DNS':
            self.create_scp_file('clean', 'tr_s')
            self.create_scp_file('noise', 'tr_n')
        elif self.dataset_name in ['WSJ0', 'WHAM', 'WHAMR', 'LIBRISPEECH', 'LIBRI_TTS_R']:
            self.create_scp_file('train', 'tr')
            self.create_scp_file('dev', 'cv')
            self.create_scp_file('test', 'tt')
        elif self.dataset_name == 'DAPS':
            self.create_scp_file('clean', 'clean')
            self.create_scp_file('noisy', 'noisy')
        else:
            print(f"Unknown dataset type: {self.dataset_name}")


def main():
    """Example usage"""
    # Create DNS dataset SCP files with relative paths
    dns_creator = SCPCreator('DNS', use_relative_paths=True)
    dns_creator.create_standard_scp_files()
    
    # Create WSJ0 dataset SCP files with relative paths
    wsj_creator = SCPCreator('WSJ0', use_relative_paths=True)
    wsj_creator.create_scp_file('cv', 'cv_s1')  # Specific subset
    
    # Create WHAM dataset SCP files with absolute paths
    wham_creator = SCPCreator('WHAM', use_relative_paths=False)
    wham_creator.create_standard_scp_files()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create SCP files for datasets using environment configuration')
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Dataset name (DNS, WSJ0, WHAM, WHAMR, LIBRISPEECH, LIBRI_TTS_R, DAPS)')
    parser.add_argument('--relative', action='store_true', default=True,
                       help='Use relative paths in SCP files (default: True)')
    parser.add_argument('--absolute', action='store_true',
                       help='Use absolute paths in SCP files')
    parser.add_argument('--subset', type=str, 
                       help='Specific subset to process (e.g., train, dev, test, clean, noise)')
    parser.add_argument('--suffix', type=str,
                       help='Output SCP file suffix (required if --subset is specified)')
    
    args = parser.parse_args()
    
    # Determine path type
    use_relative = not args.absolute if args.absolute else args.relative
    
    # Create SCP creator
    creator = SCPCreator(args.dataset, use_relative_paths=use_relative)
    
    if args.subset and args.suffix:
        # Create specific subset
        creator.create_scp_file(args.subset, args.suffix)
    else:
        # Create standard SCP files
        creator.create_standard_scp_files()