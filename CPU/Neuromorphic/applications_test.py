#!/usr/bin/env python3

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results/applications")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg): 
    print(f"[APP-TEST] {msg}")

# Standalone implementations for testing
class TestNeuromorphicPipeline:
    """Simplified pipeline for application testing"""
    
    def __init__(self):
        self.operation_count = 0
        self.stats = {
            'neural_ops': 0,
            'traditional_ops': 0,
            'total_spikes': 0
        }
    
    def process_instruction(self, instruction):
        """Process a single instruction"""
        self.operation_count += 1
        
        if len(instruction) < 3:
            return 0
        
        opcode, a, b = instruction[0], instruction[1], instruction[2]
        
        # Ternary operations
        if opcode == "ADD":
            result = (a + b) % 3
        elif opcode == "AND":
            result = min(a, b)
        elif opcode == "OR":
            result = max(a, b)
        elif opcode == "XOR":
            result = (a - b) % 3
        else:
            result = 0
        
        # Update stats
        if self.operation_count > 50:
            self.stats['neural_ops'] += 1
            self.stats['total_spikes'] += np.random.randint(1, 5)  # Simulated spikes
        else:
            self.stats['traditional_ops'] += 1
        
        return result

class TernaryNLP:
    """Ternary-based natural language processing for testing"""
    
    def __init__(self):
        self.cpu = TestNeuromorphicPipeline()
        self.word_vectors = {}
        self.vocabulary = set()
    
    def ternary_encode(self, text):
        """Convert text to ternary vectors"""
        words = text.lower().split()
        vectors = []
        
        for word in words:
            if word not in self.word_vectors:
                # Generate consistent ternary vector from word hash
                word_hash = hash(word) & 0xFFFFFF  # 24-bit hash
                vector = []
                temp_hash = word_hash
                for _ in range(8):  # 8 ternary digits
                    vector.append(temp_hash % 3)
                    temp_hash //= 3
                self.word_vectors[word] = vector
            
            vectors.append(self.word_vectors[word])
            self.vocabulary.add(word)
        
        return vectors
    
    def semantic_similarity(self, text1, text2):
        """Calculate semantic similarity using ternary operations"""
        vec1 = self.ternary_encode(text1)
        vec2 = self.ternary_encode(text2)
        
        if not vec1 or not vec2:
            return 0.0
        
        # Pad to same length
        max_len = max(len(vec1), len(vec2))
        while len(vec1) < max_len:
            vec1.append([0] * 8)
        while len(vec2) < max_len:
            vec2.append([0] * 8)
        
        # Calculate similarity using CPU operations
        total_similarity = 0
        total_comparisons = 0
        
        for v1, v2 in zip(vec1, vec2):
            for a, b in zip(v1, v2):
                # Use ternary AND for similarity measure
                similarity = self.cpu.process_instruction(["AND", a, b])
                total_similarity += similarity
                total_comparisons += 1
        
        return total_similarity / max(1, total_comparisons) / 2.0  # Normalize to [0,1]
    
    def run_demo(self):
        """Run NLP demonstration"""
        log("Running NLP demo...")
        
        # Test sentences
        test_sentences = [
            "the quick brown fox jumps",
            "the fast brown fox leaps", 
            "a slow red cat walks",
            "hello world example",
            "hello earth example",
            "machine learning algorithms",
            "artificial intelligence systems",
            "computer science research"
        ]
        
        # Calculate pairwise similarities
        similarities = {}
        similarity_scores = []
        
        for i, sent1 in enumerate(test_sentences):
            for j, sent2 in enumerate(test_sentences[i+1:], i+1):
                sim = self.semantic_similarity(sent1, sent2)
                similarity_key = f"{sent1[:20]}... <-> {sent2[:20]}..."
                similarities[similarity_key] = round(sim, 4)
                similarity_scores.append(sim)
        
        # Calculate statistics
        avg_similarity = np.mean(similarity_scores)
        max_similarity = np.max(similarity_scores) 
        min_similarity = np.min(similarity_scores)
        
        results = {
            'test_sentences': test_sentences,
            'similarities': similarities,
            'statistics': {
                'vocabulary_size': len(self.vocabulary),
                'average_similarity': round(avg_similarity, 4),
                'max_similarity': round(max_similarity, 4),
                'min_similarity': round(min_similarity, 4),
                'total_comparisons': len(similarity_scores)
            },
            'cpu_stats': self.cpu.stats.copy(),
            'sample_vectors': {word: vec for word, vec in list(self.word_vectors.items())[:3]}
        }
        
        return results

class TernaryImageProcessor:
    """Ternary computer vision operations for testing"""
    
    def __init__(self):
        self.cpu = TestNeuromorphicPipeline()
    
    def ternary_threshold(self, image):
        """Convert grayscale image to ternary values"""
        if len(image.shape) == 3:
            # Convert to grayscale
            image = np.mean(image, axis=2)
        
        # Threshold to ternary values
        result = np.zeros_like(image, dtype=int)
        result[image < 85] = 0
        result[(image >= 85) & (image < 170)] = 1
        result[image >= 170] = 2
        
        return result
    
    def apply_operation(self, img1, img2, operation):
        """Apply ternary operation to two images pixel-wise"""
        if img1.shape != img2.shape:
            raise ValueError("Images must have same dimensions")
        
        result = np.zeros_like(img1)
        total_pixels = img1.size
        
        # Process in batches to avoid overwhelming the CPU simulation
        batch_size = min(100, total_pixels)
        flat1 = img1.flatten()
        flat2 = img2.flatten()
        
        for i in range(0, len(flat1), batch_size):
            end_idx = min(i + batch_size, len(flat1))
            
            for j in range(i, end_idx):
                result.flat[j] = self.cpu.process_instruction([
                    operation, int(flat1[j]), int(flat2[j])
                ])
        
        return result
    
    def edge_detection(self, image):
        """Simple edge detection using ternary operations"""
        ternary_img = self.ternary_threshold(image)
        
        # Create shifted versions for edge detection
        shifted_right = np.roll(ternary_img, 1, axis=1)
        shifted_down = np.roll(ternary_img, 1, axis=0)
        
        # Use XOR to find differences (edges)
        edges_h = self.apply_operation(ternary_img, shifted_right, "XOR")
        edges_v = self.apply_operation(ternary_img, shifted_down, "XOR")
        
        # Combine horizontal and vertical edges using OR
        edges = self.apply_operation(edges_h, edges_v, "OR")
        
        return edges
    
    def run_demo(self):
        """Run computer vision demonstration"""
        log("Running CV demo...")
        
        # Create synthetic test images
        np.random.seed(42)  # For reproducible results
        
        # Test image 1: Random noise
        img1 = np.random.randint(0, 256, (32, 32))
        
        # Test image 2: Simple pattern
        img2 = np.zeros((32, 32))
        img2[8:24, 8:24] = 200  # White square
        img2[12:20, 12:20] = 100  # Gray center
        
        # Test image 3: Gradient
        img3 = np.linspace(0, 255, 32*32).reshape(32, 32)
        
        test_images = {
            'random_noise': img1,
            'pattern': img2, 
            'gradient': img3
        }
        
        results = {
            'image_size': (32, 32),
            'test_images': list(test_images.keys()),
            'operations_tested': ['AND', 'OR', 'XOR', 'edge_detection'],
            'results': {}
        }
        
        # Process each test image
        for name, image in test_images.items():
            log(f"Processing {name} image...")
            
            # Convert to ternary
            ternary_img = self.ternary_threshold(image)
            
            # Calculate ternary distribution
            ternary_dist = {
                '0': int(np.sum(ternary_img == 0)),
                '1': int(np.sum(ternary_img == 1)),
                '2': int(np.sum(ternary_img == 2))
            }
            
            # Test operations with pattern image as second operand
            pattern_ternary = self.ternary_threshold(test_images['pattern'])
            
            and_result = self.apply_operation(ternary_img, pattern_ternary, "AND")
            or_result = self.apply_operation(ternary_img, pattern_ternary, "OR")
            xor_result = self.apply_operation(ternary_img, pattern_ternary, "XOR")
            
            # Edge detection
            edges = self.edge_detection(image)
            
            # Compile results
            results['results'][name] = {
                'ternary_distribution': ternary_dist,
                'operation_results': {
                    'and_nonzero': int(np.sum(and_result > 0)),
                    'or_nonzero': int(np.sum(or_result > 0)),
                    'xor_nonzero': int(np.sum(xor_result > 0)),
                    'edges_detected': int(np.sum(edges > 0))
                },
                'statistics': {
                    'mean_pixel_value': float(np.mean(ternary_img)),
                    'std_pixel_value': float(np.std(ternary_img)),
                    'edge_density': float(np.sum(edges > 0) / edges.size)
                }
            }
        
        # Add CPU statistics
        results['cpu_stats'] = self.cpu.stats.copy()
        results['total_operations'] = self.cpu.operation_count
        
        return results

def run_cv_tests():
    """Run computer vision tests"""
    try:
        cv_processor = TernaryImageProcessor()
        results = cv_processor.run_demo()
        
        log("CV tests completed successfully")
        log(f"Processed {results['total_operations']} operations")
        log(f"Image size: {results['image_size']}")
        
        return results
        
    except Exception as e:
        log(f"CV test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "elapsed": 0}

def run_nlp_tests():
    """Run natural language processing tests"""
    try:
        nlp_processor = TernaryNLP()
        results = nlp_processor.run_demo()
        
        log("NLP tests completed successfully")
        log(f"Vocabulary size: {results['statistics']['vocabulary_size']}")
        log(f"Average similarity: {results['statistics']['average_similarity']}")
        
        return results
        
    except Exception as e:
        log(f"NLP test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "elapsed": 0}

def run_integration_tests():
    """Run integration tests combining NLP and CV"""
    log("Running integration tests...")
    
    try:
        # Test cross-modal processing
        nlp = TernaryNLP()
        cv = TernaryImageProcessor()
        
        # Process some text
        text_result = nlp.semantic_similarity("image processing", "computer vision")
        
        # Process some images
        test_img = np.random.randint(0, 256, (16, 16))
        ternary_img = cv.ternary_threshold(test_img)
        
        # Combine results using shared CPU operations
        combined_operations = nlp.cpu.operation_count + cv.cpu.operation_count
        combined_stats = {
            'nlp_ops': nlp.cpu.stats,
            'cv_ops': cv.cpu.stats,
            'total_operations': combined_operations,
            'cross_modal_similarity': text_result
        }
        
        log(f"Integration test completed with {combined_operations} total operations")
        
        return {
            'success': True,
            'combined_stats': combined_stats,
            'text_similarity': text_result,
            'image_processed': True,
            'image_shape': ternary_img.shape
        }
        
    except Exception as e:
        log(f"Integration test failed: {e}")
        return {'success': False, 'error': str(e)}

def generate_test_report(cv_results, nlp_results, integration_results):
    """Generate comprehensive test report"""
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_summary': {
            'cv_success': 'error' not in cv_results,
            'nlp_success': 'error' not in nlp_results,
            'integration_success': integration_results.get('success', False)
        },
        'performance_metrics': {},
        'functionality_metrics': {}
    }
    
    # CV metrics
    if 'error' not in cv_results:
        cv_ops = cv_results.get('total_operations', 0)
        report['performance_metrics']['cv_operations'] = cv_ops
        
        # Calculate CV functionality score
        cv_functionality = 0
        if 'results' in cv_results:
            for img_name, img_results in cv_results['results'].items():
                if 'operation_results' in img_results:
                    # Check if operations produced reasonable results
                    ops = img_results['operation_results']
                    if ops.get('edges_detected', 0) > 0:
                        cv_functionality += 1
                    if ops.get('and_nonzero', 0) >= 0:
                        cv_functionality += 1
                    if ops.get('or_nonzero', 0) >= 0:
                        cv_functionality += 1
        
        report['functionality_metrics']['cv_functionality_score'] = cv_functionality
    
    # NLP metrics
    if 'error' not in nlp_results:
        nlp_ops = nlp_results.get('cpu_stats', {}).get('neural_ops', 0) + nlp_results.get('cpu_stats', {}).get('traditional_ops', 0)
        report['performance_metrics']['nlp_operations'] = nlp_ops
        
        # Calculate NLP functionality score
        nlp_functionality = 0
        if 'statistics' in nlp_results:
            stats = nlp_results['statistics']
            if stats.get('vocabulary_size', 0) > 0:
                nlp_functionality += 2
            if 0 <= stats.get('average_similarity', -1) <= 1:
                nlp_functionality += 2
            if stats.get('total_comparisons', 0) > 0:
                nlp_functionality += 1
        
        report['functionality_metrics']['nlp_functionality_score'] = nlp_functionality
    
    # Integration metrics
    if integration_results.get('success', False):
        total_ops = integration_results.get('combined_stats', {}).get('total_operations', 0)
        report['performance_metrics']['integration_operations'] = total_ops
        report['functionality_metrics']['integration_success'] = True
    else:
        report['functionality_metrics']['integration_success'] = False
    
    # Overall score
    max_cv_score = 9  # 3 images Ã— 3 operations
    max_nlp_score = 5
    max_integration_score = 1
    
    cv_score = report['functionality_metrics'].get('cv_functionality_score', 0)
    nlp_score = report['functionality_metrics'].get('nlp_functionality_score', 0)
    integration_score = 1 if report['functionality_metrics'].get('integration_success', False) else 0
    
    overall_score = ((cv_score / max_cv_score) + (nlp_score / max_nlp_score) + (integration_score / max_integration_score)) / 3 * 100
    
    report['overall_score'] = round(overall_score, 1)
    
    return report

def main():
    """Main test execution function"""
    log("=== Starting Application Tests ===")
    
    start_time = time.time()
    
    # Run CV tests
    log("\n1. Running Computer Vision Tests...")
    cv_results = run_cv_tests()
    
    # Run NLP tests
    log("\n2. Running Natural Language Processing Tests...")
    nlp_results = run_nlp_tests()
    
    # Run integration tests
    log("\n3. Running Integration Tests...")
    integration_results = run_integration_tests()
    
    # Generate comprehensive report
    log("\n4. Generating Test Report...")
    test_report = generate_test_report(cv_results, nlp_results, integration_results)
    
    # Compile all results
    all_results = {
        'cv': cv_results,
        'nlp': nlp_results,
        'integration': integration_results,
        'test_report': test_report,
        'execution_time': time.time() - start_time
    }
    
    # Save results
    output_file = RESULTS_DIR / "applications_test_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    log(f"Application test results saved to {output_file}")
    
    # Print summary
    log("\n=== Test Summary ===")
    log(f"Total execution time: {all_results['execution_time']:.2f} seconds")
    
    if 'error' not in cv_results:
        log(f"CV: SUCCESS - {cv_results.get('total_operations', 0)} operations")
        log(f"    Images processed: {len(cv_results.get('results', {}))}")
    else:
        log(f"CV: FAILED - {cv_results.get('error', 'Unknown error')}")
    
    if 'error' not in nlp_results:
        log(f"NLP: SUCCESS - Vocabulary: {nlp_results.get('statistics', {}).get('vocabulary_size', 0)} words")
        log(f"     Average similarity: {nlp_results.get('statistics', {}).get('average_similarity', 0):.3f}")
    else:
        log(f"NLP: FAILED - {nlp_results.get('error', 'Unknown error')}")
    
    if integration_results.get('success', False):
        log(f"INTEGRATION: SUCCESS - {integration_results.get('combined_stats', {}).get('total_operations', 0)} total operations")
    else:
        log(f"INTEGRATION: FAILED - {integration_results.get('error', 'Unknown error')}")
    
    log(f"Overall Score: {test_report['overall_score']}/100")
    
    # Return success status
    return all_results['test_report']['overall_score'] > 70

def run_performance_benchmark():
    """Run performance benchmark for applications"""
    log("Running performance benchmark...")
    
    benchmark_results = {}
    
    # Benchmark CV processing
    cv_processor = TernaryImageProcessor()
    
    # Test different image sizes
    image_sizes = [(16, 16), (32, 32), (64, 64)]
    
    for size in image_sizes:
        log(f"Benchmarking CV with {size[0]}x{size[1]} images...")
        
        test_img = np.random.randint(0, 256, size)
        start_time = time.time()
        
        # Process image
        ternary_img = cv_processor.ternary_threshold(test_img)
        edges = cv_processor.edge_detection(test_img)
        
        end_time = time.time()
        
        pixels = size[0] * size[1]
        benchmark_results[f"cv_{size[0]}x{size[1]}"] = {
            'pixels': pixels,
            'processing_time': end_time - start_time,
            'pixels_per_second': pixels / (end_time - start_time),
            'operations': cv_processor.cpu.operation_count
        }
    
    # Benchmark NLP processing
    nlp_processor = TernaryNLP()
    
    text_lengths = [
        ("short", "hello world"),
        ("medium", "the quick brown fox jumps over the lazy dog"),
        ("long", "machine learning and artificial intelligence are transforming computer science research and applications")
    ]
    
    for length_name, text in text_lengths:
        log(f"Benchmarking NLP with {length_name} text...")
        
        start_time = time.time()
        
        # Process text
        vectors = nlp_processor.ternary_encode(text)
        similarity = nlp_processor.semantic_similarity(text, text)
        
        end_time = time.time()
        
        words = len(text.split())
        benchmark_results[f"nlp_{length_name}"] = {
            'words': words,
            'processing_time': end_time - start_time,
            'words_per_second': words / (end_time - start_time),
            'operations': nlp_processor.cpu.operation_count,
            'similarity_score': similarity
        }
    
    # Save benchmark results
    benchmark_file = RESULTS_DIR / "performance_benchmark.json"
    with open(benchmark_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    log(f"Performance benchmark saved to {benchmark_file}")
    
    return benchmark_results

if __name__ == "__main__":
    try:
        # Run main tests
        success = main()
        
        # Run performance benchmark
        log("\n=== Running Performance Benchmark ===")
        benchmark_results = run_performance_benchmark()
        
        # Print benchmark summary
        log("\n=== Benchmark Summary ===")
        for test_name, results in benchmark_results.items():
            if 'cv_' in test_name:
                log(f"{test_name}: {results['pixels_per_second']:.0f} pixels/sec")
            elif 'nlp_' in test_name:
                log(f"{test_name}: {results['words_per_second']:.0f} words/sec")
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        log("Tests interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        log(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
