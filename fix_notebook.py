import json

with open('Project.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

changed = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "Best Model: Random Forest" in source and "y_pred" in source:
            print("Found buggy cell, fixing...")
            new_source = source.replace(
                "print(f\"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}\")\n",
                "y_pred_rf = trained_models['Random Forest'].predict(X_test_scaled)\nprint(f\"Precision: {precision_score(y_test, y_pred_rf, zero_division=0):.4f}\")\n"
            )
            new_source = new_source.replace("precision_score(y_test, y_pred, zero_division=0)", "precision_score(y_test, y_pred_rf, zero_division=0)")
            new_source = new_source.replace("recall_score(y_test, y_pred, zero_division=0)", "recall_score(y_test, y_pred_rf, zero_division=0)")
            new_source = new_source.replace("f1_score(y_test, y_pred, zero_division=0)", "f1_score(y_test, y_pred_rf, zero_division=0)")
            
            # Re-split into lines for jupyter format
            lines = []
            for line in new_source.split('\n'):
                lines.append(line + '\n')
            # Remove the last extra newline if it wasn't there
            if not source.endswith('\n'):
                lines[-1] = lines[-1].rstrip('\n')
                
            cell['source'] = lines
            changed = True

if changed:
    with open('Project.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Fixed Project.ipynb")
else:
    print("No matching bug found in Project.ipynb")
