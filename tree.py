#!/usr/bin/env python3
"""
自动生成项目 README.md 文档
扫描目录结构，分析Python文件，生成完整文档
"""

import os
import ast
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple


class CodeAnalyzer:
    """代码分析器"""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.exclude_dirs = {
            '__pycache__', '.git', '.venv', 'venv', 'env',
            'node_modules', '.ipynb_checkpoints', 'build', 'dist',
            '.pytest_cache', '.mypy_cache', '*.egg-info'
        }
        self.modules = {}

    def should_skip(self, path: Path) -> bool:
        """判断是否应该跳过该路径"""
        for part in path.parts:
            if part in self.exclude_dirs or part.startswith('.'):
                return True
        return False

    def analyze_python_file(self, filepath: Path) -> Dict:
        """分析单个Python文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            functions = []
            classes = []
            imports = []
            docstring = ast.get_docstring(tree) or ""

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_doc = ast.get_docstring(node) or ""
                    # 提取参数
                    args = [arg.arg for arg in node.args.args]
                    functions.append({
                        'name': node.name,
                        'args': args,
                        'docstring': func_doc.split('\n')[0] if func_doc else ""
                    })

                elif isinstance(node, ast.ClassDef):
                    class_doc = ast.get_docstring(node) or ""
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    classes.append({
                        'name': node.name,
                        'methods': methods,
                        'docstring': class_doc.split('\n')[0] if class_doc else ""
                    })

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            return {
                'path': filepath,
                'docstring': docstring,
                'functions': functions,
                'classes': classes,
                'imports': list(set(imports)),
                'lines': len(content.split('\n'))
            }
        except Exception as e:
            print(f"警告: 无法分析 {filepath}: {e}")
            return None

    def scan_directory(self):
        """扫描目录"""
        for py_file in self.root_dir.rglob('*.py'):
            if self.should_skip(py_file):
                continue

            rel_path = py_file.relative_to(self.root_dir)
            module_name = str(rel_path.parent).replace(os.sep, '.')

            if module_name not in self.modules:
                self.modules[module_name] = []

            analysis = self.analyze_python_file(py_file)
            if analysis:
                self.modules[module_name].append(analysis)

    def generate_tree(self, max_depth: int = 3) -> str:
        """生成目录树"""

        def tree_recursive(directory: Path, prefix: str = '', depth: int = 0) -> List[str]:
            if depth >= max_depth:
                return []

            lines = []
            try:
                entries = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                entries = [e for e in entries if not any(ex in e.name for ex in self.exclude_dirs)]

                for i, entry in enumerate(entries):
                    is_last = i == len(entries) - 1
                    connector = '└── ' if is_last else '├── '

                    if entry.is_dir():
                        lines.append(f'{prefix}{connector}{entry.name}/')
                        extension = '    ' if is_last else '│   '
                        lines.extend(tree_recursive(entry, prefix + extension, depth + 1))
                    else:
                        lines.append(f'{prefix}{connector}{entry.name}')
            except PermissionError:
                pass

            return lines

        tree_lines = [f'{self.root_dir.name}/']
        tree_lines.extend(tree_recursive(self.root_dir))
        return '\n'.join(tree_lines)


class ReadmeGenerator:
    """README.md 生成器"""

    def __init__(self, analyzer: CodeAnalyzer):
        self.analyzer = analyzer
        self.readme_parts = []

    def add_header(self):
        """添加文档头部"""
        project_name = self.analyzer.root_dir.name
        self.readme_parts.append(f"""# {project_name.upper()} - Molecular Property Prediction Framework

A deep learning framework for molecular property prediction with uncertainty quantification.

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
""")

    def add_toc(self):
        """添加目录"""
        self.readme_parts.append("""## 📑 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Documentation](#module-documentation)
- [Command Line Usage](#command-line-usage)
- [API Reference](#api-reference)

---
""")

    def add_overview(self):
        """添加项目概述"""
        total_files = sum(len(files) for files in self.analyzer.modules.values())
        total_lines = sum(
            f['lines'] for files in self.analyzer.modules.values()
            for f in files if f
        )

        self.readme_parts.append(f"""## 🎯 Overview

This project contains **{total_files} Python files** with approximately **{total_lines:,} lines of code**.

### Key Features
- Message Passing Neural Networks for molecular graphs
- Evidential regression for uncertainty quantification
- PFAS-specific feature extraction
- Multiple confidence estimation methods
- Cross-validation and ensemble training

---
""")

    def add_project_structure(self):
        """添加项目结构"""
        tree = self.analyzer.generate_tree(max_depth=3)
        self.readme_parts.append(f"""## 📂 Project Structure
```
{tree}
```

---
""")

    def add_installation(self):
        """添加安装说明"""
        self.readme_parts.append("""## 💻 Installation

### Prerequisites
```bash
# Python 3.7+
python --version

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate  # Windows
```

### Install Dependencies
```bash
pip install torch rdkit-pypi numpy pandas scikit-learn
pip install tensorboardX tqdm
```

### Install Package
```bash
# Development mode
pip install -e .

# Or directly
python setup.py install
```

---
""")

    def add_quick_start(self):
        """添加快速开始"""
        self.readme_parts.append("""## 🚀 Quick Start

### 1. Prepare Your Data
```bash
# CSV format: smiles, target_value
cat > data/train.csv << EOF
smiles,target
CCO,1.23
c1ccccc1,2.45
CC(C)O,0.89
EOF
```

### 2. Train a Model
```bash
python train.py \\
    --data_path data/train.csv \\
    --save_dir results/my_model \\
    --epochs 30 \\
    --hidden_size 300 \\
    --depth 3
```

### 3. Make Predictions
```bash
python predict.py \\
    --test_path data/test.csv \\
    --checkpoint_dir results/my_model/fold_0 \\
    --preds_path predictions/output.csv
```

---
""")

    def add_module_docs(self):
        """添加模块文档"""
        self.readme_parts.append("## 📚 Module Documentation\n\n")

        # 按模块分类
        sorted_modules = sorted(self.analyzer.modules.items())

        for module_path, files in sorted_modules:
            if not files or module_path == '.':
                continue

            # 模块标题
            module_name = module_path.replace('.', ' / ').title()
            self.readme_parts.append(f"### {module_name}\n\n")

            for file_info in files:
                if not file_info:
                    continue

                filename = file_info['path'].name
                self.readme_parts.append(f"#### **`{filename}`**\n")

                if file_info['docstring']:
                    self.readme_parts.append(f"{file_info['docstring']}\n\n")

                # 列出类
                if file_info['classes']:
                    self.readme_parts.append("**Classes:**\n")
                    for cls in file_info['classes'][:5]:  # 最多显示5个
                        self.readme_parts.append(f"- `{cls['name']}` - {cls['docstring']}\n")
                    self.readme_parts.append("\n")

                # 列出主要函数
                if file_info['functions']:
                    self.readme_parts.append("**Key Functions:**\n")
                    for func in file_info['functions'][:5]:  # 最多显示5个
                        args_str = ', '.join(func['args'][:3])  # 最多显示3个参数
                        if len(func['args']) > 3:
                            args_str += ', ...'
                        self.readme_parts.append(
                            f"- `{func['name']}({args_str})` - {func['docstring']}\n"
                        )
                    self.readme_parts.append("\n")

                self.readme_parts.append("---\n\n")

    def add_cli_usage(self):
        """添加命令行使用说明"""
        self.readme_parts.append("""## 🔧 Command Line Usage

### Training Commands

#### Basic Training
```bash
python train.py \\
    --data_path data/molecules.csv \\
    --save_dir results/basic_model \\
    --epochs 30 \\
    --batch_size 50
```

#### With PFAS Features
```bash
python train.py \\
    --data_path data/pfas_molecules.csv \\
    --save_dir results/pfas_model \\
    --features_generator pfas \\
    --epochs 50
```

#### Evidential Regression (with Uncertainty)
```bash
python train.py \\
    --data_path data/molecules.csv \\
    --save_dir results/evidential \\
    --confidence evidence \\
    --new_loss \\
    --regularizer_coeff 1.0 \\
    --temperature 1.0 \\
    --use_adaptive_reg \\
    --use_robust_loss
```

#### Cross-Validation
```bash
python train.py \\
    --data_path data/molecules.csv \\
    --save_dir results/cv_experiment \\
    --num_folds 5 \\
    --epochs 30
```

### Prediction Commands

#### Basic Prediction
```bash
python predict.py \\
    --test_path data/test.csv \\
    --checkpoint_dir results/basic_model/fold_0 \\
    --preds_path predictions/test_predictions.csv
```

#### Ensemble Prediction
```bash
python predict.py \\
    --test_path data/test.csv \\
    --checkpoint_dir results/basic_model \\
    --preds_path predictions/ensemble_predictions.csv
```

### Parameter Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_path` | Path to training data CSV | Required |
| `--save_dir` | Directory to save models | Required |
| `--epochs` | Number of training epochs | 30 |
| `--batch_size` | Batch size | 50 |
| `--hidden_size` | Hidden layer dimension | 300 |
| `--depth` | Message passing depth | 3 |
| `--dropout` | Dropout probability | 0.0 |
| `--confidence` | Uncertainty method | None |
| `--features_generator` | Feature generator | None |

---
""")

    def add_api_reference(self):
        """添加API参考"""
        self.readme_parts.append("""## 📖 API Reference

### Python API

#### Training
```python
from chemprop.train import run_training
from chemprop.data import get_dataset_splits

# Load data
(train, val, test), features_scaler, scaler = get_dataset_splits(
    data_path='data/molecules.csv',
    args=args,
    logger=logger
)

# Train model
models = run_training(
    train_data=train,
    val_data=val,
    scaler=scaler,
    features_scaler=features_scaler,
    args=args,
    logger=logger
)
```

#### Prediction
```python
from chemprop.train import predict, make_predictions

# Method 1: Using trained model
predictions = predict(
    model=model,
    data=test_data,
    batch_size=50,
    scaler=scaler
)

# Method 2: From checkpoint
predictions = make_predictions(
    args=args,
    smiles=['CCO', 'c1ccccc1']
)
```

#### Feature Generation
```python
from chemprop.features import pfas_features_generator

# Generate PFAS features
features = pfas_features_generator('FC(F)(F)C(F)(F)S(=O)(=O)O')
print(f"Feature vector length: {len(features)}")
```

---
""")

    def add_footer(self):
        """添加页脚"""
        self.readme_parts.append("""## 📝 Notes

- This documentation was automatically generated
- For detailed parameter descriptions, run `python train.py --help`
- Check individual module docstrings for more information

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

---

**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

    def generate(self) -> str:
        """生成完整文档"""
        self.add_header()
        self.add_toc()
        self.add_overview()
        self.add_project_structure()
        self.add_installation()
        self.add_quick_start()
        self.add_module_docs()
        self.add_cli_usage()
        self.add_api_reference()
        self.add_footer()

        return '\n'.join(self.readme_parts)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate README.md for ChemProp project')
    parser.add_argument('--dir', '-d', default='.', help='Project directory (default: current)')
    parser.add_argument('--output', '-o', default='README.md', help='Output file (default: README.md)')
    parser.add_argument('--depth', type=int, default=3, help='Tree depth (default: 3)')

    args = parser.parse_args()

    print(f"🔍 Analyzing project in: {args.dir}")

    # 分析代码
    analyzer = CodeAnalyzer(args.dir)
    analyzer.scan_directory()

    print(f"📊 Found {len(analyzer.modules)} modules")

    # 生成文档
    print(f"📝 Generating README...")
    generator = ReadmeGenerator(analyzer)
    readme_content = generator.generate()

    # 写入文件
    output_path = Path(args.output)
    output_path.write_text(readme_content, encoding='utf-8')

    print(f"✅ README generated: {output_path.absolute()}")
    print(f"📄 File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    main()