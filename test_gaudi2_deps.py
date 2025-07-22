#!/usr/bin/env python3
"""
Test Gaudi2 installation and dependencies
"""

def test_dependency(module_name, import_statement):
    """Test if a dependency can be imported"""
    try:
        exec(import_statement)
        print(f"✅ {module_name} available")
        return True
    except ImportError as e:
        print(f"❌ {module_name} not installed: {e}")
        return False

def main():
    print("🧪 Testing Gaudi2 Python dependencies...")
    
    tests = [
        ("optimum[habana]", "import optimum.habana"),
        ("habana_torch_plugin", "import habana_torch_plugin"),
        ("torch", "import torch"),
        ("transformers", "import transformers"),
        ("Gaudi2LLM", "from gaudi2_llm import Gaudi2LLM")
    ]
    
    passed = 0
    for name, import_cmd in tests:
        if test_dependency(name, import_cmd):
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} dependencies available")
    
    if passed == len(tests):
        print("🎉 All Gaudi2 dependencies are ready!")
    else:
        print("⚠️  Some dependencies missing. Run: make install-gaudi2-deps")

if __name__ == "__main__":
    main()
