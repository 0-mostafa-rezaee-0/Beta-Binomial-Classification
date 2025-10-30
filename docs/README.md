<h1 align="center">Documentation Directory</h1>

This directory contains comprehensive documentation materials for the IRT (Item Response Theory) project, including explanatory content, multimedia resources, and references to generated visualizations.

## Directory Contents

### 📊 Visualizations (Symbolic Links)
- `adaptive_testing_convergence.png` → Links to `../figures/adaptive_testing_convergence.png`
- `item_characteristic_curves.png` → Links to `../figures/item_characteristic_curves.png`

*Note: These are symbolic links to the actual figures stored in the `figures/` directory. This approach avoids duplication while making visualizations available for documentation purposes.*

### 📚 Documentation Files
- `IRT-Conceptual-Overview.md` - Comprehensive conceptual overview of Item Response Theory
- `IRT-Technical-Notes.pdf` - Detailed technical notes and mathematical formulations
- `README.md` - This file explaining the documentation structure

### 🎥 Multimedia Resources
- `IRT-Video-Presentation.mp4` - Video presentation covering IRT fundamentals
- `IRT-Audio-Conversation.m4a` - Audio conversation discussing IRT applications

## Directory Structure

```
docs/
├── README.md                           # This documentation file
├── IRT-Conceptual-Overview.md         # Conceptual overview of IRT
├── IRT-Technical-Notes.pdf             # Technical documentation
├── IRT-Video-Presentation.mp4         # Video presentation
├── IRT-Audio-Conversation.m4a         # Audio conversation
├── adaptive_testing_convergence.png    # → ../figures/ (symbolic link)
└── item_characteristic_curves.png      # → ../figures/ (symbolic link)
```

## Organization Philosophy

This directory follows a clear separation of concerns:

1. **Generated Content**: All code-generated visualizations are stored in the `figures/` directory
2. **Documentation**: This directory contains explanatory materials and references to figures
3. **No Duplication**: Symbolic links prevent file duplication while maintaining accessibility
4. **Clear Purpose**: Each directory has a specific role in the project structure

## Usage Guidelines

- **For Code Outputs**: Check the `figures/` directory for all generated visualizations
- **For Documentation**: Use this directory for explanatory content and educational materials
- **For References**: Use symbolic links to reference figures in documentation without duplication

## Integration with Project Structure

This documentation directory works in harmony with:
- `figures/` - Contains all generated visualizations
- `src/` - Contains source code that generates the figures
- `data/` - Contains datasets used in analysis
- `notebooks/` - Contains interactive analysis notebooks

The symbolic link approach ensures that documentation can reference visualizations while maintaining a clean separation between generated content and explanatory materials.
