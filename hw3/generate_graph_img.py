import os
from src.graph import build_graph

def export_visualization():
    print("Generating graph visualization...")
    graph = build_graph()
    
    try:
        # Use LangGraph's built-in mermaid rendering capability
        png_data = graph.get_graph().draw_mermaid_png()
        
        # Ensure path target directory exists
        os.makedirs("results", exist_ok=True)
        
        with open("results/graph.png", "wb") as f:
            f.write(png_data)
        print("✅ Graph visualization successfully saved to: 'results/graph.png'")
    except Exception as e:
        print(f"❌ Failed to generate image file automatically: {str(e)}")
        print("Note: Renders often require graphviz or an active internet connection for the PyPI wrappers.")

if __name__ == "__main__":
    export_visualization()