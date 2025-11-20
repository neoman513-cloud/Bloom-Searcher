#!/usr/bin/env python3
"""
Complete test to verify Python and CUDA bloom filter compatibility
Optimized for minimal false positive rate with high-performance processing
"""

import hashlib
import base58
import bech32
from bitarray import bitarray
import sys
import math
from multiprocessing import Pool, cpu_count
import time

NUM_HASH_FUNCTIONS = 10
FNV_OFFSET = 0xcbf29ce484222325
FNV_PRIME  = 0x100000001b3
GOLDEN64   = 0x9e3779b97f4a7c13

def fnv_hash(data: bytes, seed: int) -> int:
    h = FNV_OFFSET ^ seed
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h

def address_to_hash160(address: str) -> bytes | None:
    """Convert Bitcoin address to 20-byte hash160"""
    try:
        if address.startswith('1') or address.startswith('3'):
            decoded = base58.b58decode_check(address)
            return decoded[1:]
        elif address.startswith('bc1'):
            hrp, data = bech32.bech32_decode(address)
            if data is None:
                return None
            decoded = bech32.convertbits(data[1:], 5, 8, False)
            if decoded is None:
                return None
            return bytes(decoded)
        else:
            return None
    except Exception as e:
        print(f"Error decoding address: {e}")
        return None

def load_addresses_from_file(filename: str) -> list[str]:
    """Load addresses from a text file (one address per line)"""
    addresses = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                addr = line.strip()
                # Skip empty lines and comments
                if addr and not addr.startswith('#'):
                    addresses.append(addr)
        print(f"Loaded {len(addresses):,} addresses from {filename}")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    return addresses

def calculate_optimal_parameters(n: int, target_fpr: float = None, max_size_mb: float = None):
    """
    Calculate optimal bloom filter parameters
    
    Args:
        n: Number of elements
        target_fpr: Target false positive rate (if None, uses minimal practical FPR)
        max_size_mb: Maximum size constraint in MB (if None, no limit)
    
    Returns:
        (bits, bytes, size_mb, k, fpr)
    """
    # If no target FPR specified, use a balanced rate
    if target_fpr is None:
        # Use 0.001 (0.1%) as default - good balance between size and accuracy
        target_fpr = 0.001
        print(f"  Using default false positive rate: {target_fpr} ({target_fpr*100}%)")
    
    k = NUM_HASH_FUNCTIONS
    
    # For given k and p, optimal m is: m = -n*ln(p) / (ln(2)^2)
    # But we need to adjust for actual k being used
    # More accurate: m = -(n * ln(p)) / (k * ln(0.5)^2)
    # Simplified optimal formula for k hash functions:
    m_optimal = -(n * math.log(target_fpr)) / (math.log(2) ** 2)
    
    # Round up to nearest byte boundary
    bloom_bits = int(math.ceil(m_optimal))
    bloom_bytes = (bloom_bits + 7) // 8
    bloom_bits = bloom_bytes * 8  # Adjust to byte boundary
    
    size_mb = bloom_bytes / (1024 * 1024)
    
    # Apply max size constraint if specified
    if max_size_mb is not None and size_mb > max_size_mb:
        print(f"  Warning: Optimal size ({size_mb:.2f} MB) exceeds max ({max_size_mb:.2f} MB)")
        size_mb = max_size_mb
        bloom_bytes = int(size_mb * 1024 * 1024)
        bloom_bits = bloom_bytes * 8
        print(f"  Using constrained size: {size_mb:.2f} MB")
    
    # Calculate actual false positive rate with chosen size
    # Formula: p ≈ (1 - e^(-kn/m))^k
    actual_fpr = (1 - math.exp(-k * n / bloom_bits)) ** k
    
    return bloom_bits, bloom_bytes, size_mb, k, actual_fpr

def process_address_batch(args):
    """Process a batch of addresses and return bit indices to set"""
    addresses, bloom_bits = args
    bit_indices = set()
    failed = 0
    
    for addr in addresses:
        h160 = address_to_hash160(addr)
        if not h160 or len(h160) != 20:
            failed += 1
            continue
        
        for j in range(NUM_HASH_FUNCTIONS):
            seed = (j * GOLDEN64) & 0xFFFFFFFFFFFFFFFF
            hv = fnv_hash(h160, seed)
            bit_idx = hv % bloom_bits
            bit_indices.add(bit_idx)
    
    return bit_indices, failed

def create_bloom_from_file(address_file: str, output_file: str, size_mb: float = None, target_fpr: float = None, max_size_mb: float = None, num_workers: int = None):
    """
    Create bloom filter from address file with optimal sizing
    
    Args:
        address_file: Input file with addresses (one per line)
        output_file: Output bloom filter binary file
        size_mb: Fixed size in MB (if specified, overrides optimization)
        target_fpr: Target false positive rate (if None, uses minimal practical rate)
        max_size_mb: Maximum size constraint in MB
        num_workers: Number of parallel workers (default: CPU count)
    """
    start_time = time.time()
    
    addresses = load_addresses_from_file(address_file)
    n = len(addresses)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"\n{'='*70}")
    print("BLOOM FILTER OPTIMIZATION")
    print(f"{'='*70}")
    print(f"  Using {num_workers} parallel workers")
    
    # If size specified, use it; otherwise calculate optimal
    if size_mb is not None:
        bloom_bytes = int(size_mb * 1024 * 1024)
        bloom_bits = bloom_bytes * 8
        k = NUM_HASH_FUNCTIONS
        
        # Calculate FPR for given size
        actual_fpr = (1 - math.exp(-k * n / bloom_bits)) ** k
        
        print(f"  Using specified size: {size_mb:.2f} MB")
    else:
        bloom_bits, bloom_bytes, size_mb, k, actual_fpr = calculate_optimal_parameters(
            n, target_fpr, max_size_mb
        )
    
    print(f"\nBloom Filter Configuration:")
    print(f"  Elements (n): {n:,}")
    print(f"  Hash functions (k): {k}")
    print(f"  Size (m): {bloom_bits:,} bits ({bloom_bytes:,} bytes, {size_mb:.2f} MB)")
    print(f"  Bits per element: {bloom_bits/n:.2f}")
    print(f"  False positive rate: {actual_fpr:.10f} ({actual_fpr*100:.8f}%)")
    print(f"  True positive rate: 100% (guaranteed)")
    
    # Show comparison with different rates
    print(f"\n{'='*70}")
    print("COMPARISON WITH OTHER FALSE POSITIVE RATES:")
    print(f"{'='*70}")
    for comparison_fpr in [0.1, 0.01, 0.001, 0.0001]:
        comp_bits, comp_bytes, comp_size_mb, comp_k, comp_actual_fpr = calculate_optimal_parameters(n, comparison_fpr)
        print(f"  {comparison_fpr*100:7.4f}% FPR → {comp_size_mb:8.2f} MB (actual: {comp_actual_fpr*100:.6f}%)")
    print(f"{'='*70}")
    
    bloom = bitarray(bloom_bits, endian='big')
    bloom.setall(0)
    
    print(f"\nInserting addresses using parallel processing...")
    
    # Split addresses into batches for parallel processing
    batch_size = max(1000, len(addresses) // (num_workers * 4))
    batches = []
    for i in range(0, len(addresses), batch_size):
        batch = addresses[i:i + batch_size]
        batches.append((batch, bloom_bits))
    
    print(f"  Processing {len(addresses):,} addresses in {len(batches)} batches...")
    print(f"  Batch size: ~{batch_size:,} addresses")
    
    # Process batches in parallel
    all_bit_indices = set()
    total_failed = 0
    
    with Pool(processes=num_workers) as pool:
        batch_count = 0
        for bit_indices, failed in pool.imap_unordered(process_address_batch, batches):
            all_bit_indices.update(bit_indices)
            total_failed += failed
            batch_count += 1
            
            if batch_count % 10 == 0 or batch_count == len(batches):
                processed = min(batch_count * batch_size, len(addresses))
                print(f"  Progress: {processed:,}/{len(addresses):,} ({processed*100/len(addresses):.1f}%)")
    
    # Set all bits at once
    print(f"  Setting {len(all_bit_indices):,} unique bit positions...")
    for bit_idx in all_bit_indices:
        bloom[bit_idx] = 1
    
    added = len(addresses) - total_failed
    
    elapsed = time.time() - start_time
    print(f"\n✓ Successfully added: {added:,} addresses")
    if total_failed > 0:
        print(f"✗ Failed to decode: {total_failed} addresses")
    print(f"⏱ Processing time: {elapsed:.2f} seconds ({len(addresses)/elapsed:,.0f} addresses/sec)")
    
    # Calculate actual bit density
    bits_set = bloom.count(1)
    density = bits_set / bloom_bits * 100
    print(f"\nBloom filter statistics:")
    print(f"  Bits set: {bits_set:,} / {bloom_bits:,} ({density:.2f}%)")
    print(f"  Expected density: ~{(1 - math.exp(-k*added/bloom_bits))*100:.2f}%")
    
    with open(output_file, 'wb') as f:
        bloom.tofile(f)
    
    print(f"\n✓ Bloom filter saved to: {output_file}")
    print(f"  File size: {bloom_bytes:,} bytes ({size_mb:.2f} MB)")
    
    return bloom

def verify_in_bloom(bloom_file: str, addresses: list[str], verbose: bool = True):
    """Verify addresses can be found in the bloom filter"""
    bloom = bitarray(endian='big')
    with open(bloom_file, 'rb') as f:
        bloom.fromfile(f)
    
    bloom_bits = len(bloom)
    print(f"\nVerifying bloom filter: {bloom_file}")
    print(f"  Size: {bloom_bits:,} bits ({len(bloom)//8:,} bytes)")
    
    found_count = 0
    not_found_count = 0
    
    for addr in addresses:
        h160 = address_to_hash160(addr)
        if not h160:
            print(f"✗ {addr}: Invalid address")
            continue
        
        if verbose:
            print(f"\nChecking: {addr}")
            print(f"Hash160: {h160.hex()}")
        
        found = True
        for i in range(NUM_HASH_FUNCTIONS):
            seed = (i * GOLDEN64) & 0xFFFFFFFFFFFFFFFF
            hv = fnv_hash(h160, seed)
            bit_idx = hv % bloom_bits
            byte_idx = bit_idx // 8
            bit_off = bit_idx % 8
            bit_set = bloom[bit_idx]
            
            if verbose:
                print(f"  Hash#{i}: seed={seed:016x}, hash={hv:016x}, bit_idx={bit_idx:,}, "
                      f"byte={byte_idx:,}, bit_off={bit_off}, bit_set={bit_set}")
            
            if not bit_set:
                found = False
        
        if found:
            found_count += 1
            if verbose:
                print(f"  Result: ✓ FOUND")
        else:
            not_found_count += 1
            if verbose:
                print(f"  Result: ✗ NOT FOUND")
    
    print(f"\n=== Summary ===")
    print(f"Found: {found_count}/{len(addresses)}")
    print(f"Not found: {not_found_count}/{len(addresses)}")

def print_usage():
    print("Usage:")
    print("  Create bloom with optimal size (minimal FPR):")
    print("    python bloom.py create <addresses.txt> <output.bin>")
    print("    Example: python bloom.py create addresses.txt bloom.bin")
    print()
    print("  Create bloom with custom FPR:")
    print("    python bloom.py create <addresses.txt> <output.bin> --fpr <rate>")
    print("    Example: python bloom.py create addresses.txt bloom.bin --fpr 0.01")
    print()
    print("  Create bloom with size limit:")
    print("    python bloom.py create <addresses.txt> <output.bin> --max-size <MB>")
    print("    Example: python bloom.py create addresses.txt bloom.bin --max-size 500")
    print()
    print("  Create bloom with fixed size:")
    print("    python bloom.py create <addresses.txt> <output.bin> --size <MB>")
    print("    Example: python bloom.py create addresses.txt bloom.bin --size 100")
    print()
    print("  Control parallel processing:")
    print("    python bloom.py create <addresses.txt> <output.bin> --workers <N>")
    print("    Example: python bloom.py create addresses.txt bloom.bin --workers 8")
    print()
    print("  Debug mode - show hash calculations:")
    print("    python bloom.py debug <address>")
    print("    Example: python bloom.py debug 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")
    print()
    print("  Verify addresses in bloom:")
    print("    python bloom.py verify <bloom.bin> <addresses.txt>")
    print()
    print("  Test specific addresses:")
    print("    python bloom.py test <bloom.bin> <address1> [address2] ...")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create":
        if len(sys.argv) < 4:
            print("Error: Missing arguments for create command")
            print_usage()
            sys.exit(1)
        
        address_file = sys.argv[2]
        output_file = sys.argv[3]
        
        # Parse optional arguments
        size_mb = None
        target_fpr = None
        max_size_mb = None
        num_workers = None
        
        i = 4
        while i < len(sys.argv):
            if sys.argv[i] == "--fpr" and i + 1 < len(sys.argv):
                target_fpr = float(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--size" and i + 1 < len(sys.argv):
                size_mb = float(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--max-size" and i + 1 < len(sys.argv):
                max_size_mb = float(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--workers" and i + 1 < len(sys.argv):
                num_workers = int(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        
        print("=" * 70)
        print("CREATING OPTIMAL BLOOM FILTER")
        print("=" * 70)
        
        create_bloom_from_file(address_file, output_file, size_mb, target_fpr, max_size_mb, num_workers)
        
        print("\n" + "=" * 70)
        print("TESTING - Verifying first 5 addresses...")
        print("=" * 70)
        
        # Test first few addresses
        test_addrs = load_addresses_from_file(address_file)[:5]
        verify_in_bloom(output_file, test_addrs, verbose=True)
        
        print("\n" + "=" * 70)
        print("NEXT STEP: Test with CUDA")
        print("=" * 70)
        print(f"\nTo test with CUDA, get a hash160 from above and run:")
        print(f"  ./main.exe test {output_file} <hash160>")
    
    elif command == "verify":
        if len(sys.argv) < 4:
            print("Error: Missing arguments for verify command")
            print_usage()
            sys.exit(1)
        
        bloom_file = sys.argv[2]
        address_file = sys.argv[3]
        
        addresses = load_addresses_from_file(address_file)
        verify_in_bloom(bloom_file, addresses, verbose=False)
    
    elif command == "test":
        if len(sys.argv) < 4:
            print("Error: Missing arguments for test command")
            print_usage()
            sys.exit(1)
        
        bloom_file = sys.argv[2]
        test_addresses = sys.argv[3:]
        
        verify_in_bloom(bloom_file, test_addresses, verbose=True)
    
    elif command == "debug":
        if len(sys.argv) < 3:
            print("Error: Missing address for debug command")
            print_usage()
            sys.exit(1)
        
        address = sys.argv[2]
        print(f"\n{'='*70}")
        print(f"DEBUG MODE - Address Hash Analysis")
        print(f"{'='*70}")
        print(f"Address: {address}")
        
        h160 = address_to_hash160(address)
        if not h160:
            print("✗ Failed to decode address")
            sys.exit(1)
        
        print(f"Hash160 (hex): {h160.hex()}")
        print(f"Hash160 (bytes): {len(h160)} bytes")
        print(f"\nHash function calculations:")
        print(f"NUM_HASH_FUNCTIONS: {NUM_HASH_FUNCTIONS}")
        print(f"FNV_OFFSET: 0x{FNV_OFFSET:016x}")
        print(f"FNV_PRIME:  0x{FNV_PRIME:016x}")
        print(f"GOLDEN64:   0x{GOLDEN64:016x}")
        
        for i in range(NUM_HASH_FUNCTIONS):
            seed = (i * GOLDEN64) & 0xFFFFFFFFFFFFFFFF
            hv = fnv_hash(h160, seed)
            print(f"\nHash #{i}:")
            print(f"  seed = 0x{seed:016x}")
            print(f"  hash = 0x{hv:016x} ({hv})")
            
            # Show bit indices for different bloom sizes
            for test_size in [10, 100, 1000]:
                test_bits = test_size * 1024 * 1024 * 8
                bit_idx = hv % test_bits
                print(f"  bit_idx ({test_size}MB) = {bit_idx:,}")
    
    else:
        print(f"Error: Unknown command '{command}'")
        print_usage()
        sys.exit(1)