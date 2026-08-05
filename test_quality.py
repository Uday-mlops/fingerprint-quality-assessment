import os
import pandas as pd

from quality_assessment import quality_gate


IMAGE_FOLDERS = {

    "good": "images/good",

    "blur": "images/blur",

    "dark": "images/dark",

    "glare": "images/glare"
}


results = []


for category, folder in IMAGE_FOLDERS.items():

    if not os.path.exists(folder):

        print(f"Folder not found: {folder}")
        continue

    files = os.listdir(folder)

    for file_name in files:

        image_path = os.path.join(
            folder,
            file_name
        )

        try:

            result = quality_gate(image_path)

            row = {

                "Category": category,

                "Image": file_name,

                "Passed": result["passed"],

                "Score": round(
                    result["composite_score"],
                    2
                ),

                "Blur": round(
                    result["blur"]["blur_score"],
                    2
                ),

                "Brightness": round(
                    result["brightness"]["brightness"],
                    2
                ),

                "Glare": round(
                    result["glare"]["glare_fraction"],
                    4
                ),

                "ROI": round(
                    result["roi"]["roi_fraction"],
                    2
                ),

                "Ridge": round(
                    result["ridge"]["ridge_score"],
                    2
                ),

                "Guidance": result["guidance"]
            }

            results.append(row)

        except Exception as e:

            print(f"Error processing {file_name}")

            print(e)


df = pd.DataFrame(results)

print("\n")
print("=" * 120)
print(df)
print("=" * 120)

df.to_csv(
    "quality_results.csv",
    index=False
)

print("\nResults saved to quality_results.csv")