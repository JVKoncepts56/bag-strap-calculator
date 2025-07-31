import streamlit as st
import numpy as np
from scipy.special import ellipeinc
from scipy.optimize import minimize_scalar
import pandas as pd

def calculate_strap_drop_simple(strap_length, bag_width):
    """
    Calculate the drop length of a bag strap using elliptical arc equation.
    
    Parameters:
    - strap_length: Total arc length of the strap
    - bag_width: Width of the bag (distance between strap attachment points)
    
    Returns:
    - drop_height: Vertical drop distance from attachment points to lowest point of strap
    """
    
    # Half the bag width (distance from center to attachment point)
    half_width = bag_width / 2
    
    # Half the strap length (length of one side of the strap)
    half_strap = strap_length / 2
    
    # The strap must be at least as long as the bag width
    if strap_length < bag_width:
        raise ValueError("Strap length must be at least as long as the bag width")
    
    # Target full strap length for comparison
    target_full_strap_length = strap_length
    
    def objective(drop_height):
        """Objective function to minimize: difference between calculated and actual arc length"""
        if drop_height <= 0:
            return float('inf')
        
        # For an ellipse with the strap attachment points at (±half_width, 0)
        # and the lowest point at (0, -drop_height)
        a = drop_height
        b = half_width
        
        # For a quarter ellipse from bottom to side
        k = np.sqrt(1 - (min(a,b)/max(a,b))**2)
        
        # Complete elliptic integral for quarter ellipse
        if a >= b:
            quarter_arc = a * ellipeinc(np.pi/2, k**2)
        else:
            quarter_arc = b * ellipeinc(np.pi/2, k**2)
        
        # Full strap length is 2 quarter arcs (half ellipse)
        calculated_full_length = 2 * quarter_arc
        
        # Compare full lengths to full lengths!
        return abs(calculated_full_length - target_full_strap_length)
    
    # Use optimization to find the drop height that gives us the desired strap length
    result = minimize_scalar(objective, bounds=(0.1, strap_length), method='bounded')
    
    if result.success:
        drop_height = result.x
    else:
        # Fallback: use catenary approximation
        def catenary_objective(a):
            if a <= 0:
                return float('inf')
            return abs(a * np.sinh(half_width / a) - half_strap)
        
        cat_result = minimize_scalar(catenary_objective, bounds=(0.1, strap_length), method='bounded')
        if cat_result.success:
            a = cat_result.x
            drop_height = a * (np.cosh(half_width / a) - 1)
        else:
            # Ultimate fallback: parabolic approximation
            if half_strap > half_width:
                drop_height = half_width * np.sqrt(3 * (half_strap/half_width - 1) / 8)
            else:
                drop_height = 0
    
    return drop_height

# Streamlit app
st.set_page_config(page_title="Bag Strap Drop Calculator - Offset Method", page_icon="👜", layout="wide")

st.title("👜 Bag Strap Drop Calculator - Width Offset Analysis")
st.markdown("Calculate strap drop using elliptical arc equations with width offsets")

# Sidebar for instructions
with st.sidebar:
    st.header("📖 Instructions")
    st.markdown("""
    This calculator uses the elliptical arc length equation to calculate strap drop.
    
    **Three calculations are performed:**
    1. **Squared Up**: Uses the actual bag width
    2. **1 Inch Offset**: Subtracts 1" from bag width
    3. **2 Inch Offset**: Subtracts 2" from bag width
    
    **Requirements:**
    - Strap length must be greater than bag width
    - For offsets, strap length must be greater than (bag width - offset)
    
    **CSV Format:**
    ```
    Description,Strap Length,Bag Width
    Large Tote,19.0,15.75
    Small Purse,12.5,8.0
    ```
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔧 Single Bag Calculator")
    
    # Initialize session state for storing calculations
    if 'calculations' not in st.session_state:
        st.session_state.calculations = []
    
    # Input form
    with st.form("calculator_form"):
        col_desc, col_strap, col_width = st.columns([2, 1, 1])
        
        with col_desc:
            description = st.text_input("Bag Description", placeholder="e.g., Large Tote")
        with col_strap:
            strap_length = st.number_input("Strap Length (inches)", min_value=0.1, value=19.0, step=0.25)
        with col_width:
            bag_width = st.number_input("Bag Width (inches)", min_value=0.1, value=15.75, step=0.25)
        
        submitted = st.form_submit_button("Calculate", use_container_width=True)
        
        if submitted:
            results = {}
            errors = []
            
            # Calculate for different offsets
            offsets = {
                "Squared Up": 0,
                "1 Inch Offset": 1,
                "2 Inch Offset": 2
            }
            
            for name, offset in offsets.items():
                effective_width = bag_width - offset
                
                if effective_width <= 0:
                    results[name] = "N/A (width ≤ 0)"
                    errors.append(f"{name}: Effective width must be positive")
                elif strap_length < effective_width:
                    results[name] = "N/A (strap < width)"
                    errors.append(f"{name}: Strap length must be > {effective_width:.2f}\"")
                else:
                    try:
                        drop = calculate_strap_drop_simple(strap_length, effective_width)
                        results[name] = drop
                    except Exception as e:
                        results[name] = f"Error"
                        errors.append(f"{name}: {str(e)}")
            
            # Display any errors
            if errors:
                for error in errors:
                    st.warning(f"⚠️ {error}")
            
            # Add to history
            st.session_state.calculations.append({
                'Description': description or f"Bag #{len(st.session_state.calculations) + 1}",
                'Strap Length': strap_length,
                'Bag Width': bag_width,
                'Squared Up': f"{results['Squared Up']:.2f}\"" if isinstance(results['Squared Up'], (int, float)) else results['Squared Up'],
                '1 Inch Offset': f"{results['1 Inch Offset']:.2f}\"" if isinstance(results['1 Inch Offset'], (int, float)) else results['1 Inch Offset'],
                '2 Inch Offset': f"{results['2 Inch Offset']:.2f}\"" if isinstance(results['2 Inch Offset'], (int, float)) else results['2 Inch Offset']
            })
            
            if not errors:
                st.success("✅ Calculation added!")
            
            # Display current result
            st.markdown("### Current Result:")
            result_cols = st.columns(3)
            for i, (name, drop) in enumerate(results.items()):
                with result_cols[i]:
                    if isinstance(drop, (int, float)):
                        st.metric(name, f"{drop:.2f}\"")
                    else:
                        st.metric(name, drop)

with col2:
    st.subheader("📊 Calculation History")
    
    if st.session_state.calculations:
        # Clear history button
        if st.button("Clear History", type="secondary"):
            st.session_state.calculations = []
            st.rerun()
        
        # Display history as dataframe
        df = pd.DataFrame(st.session_state.calculations)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="bag_strap_calculations.csv",
            mime="text/csv"
        )
    else:
        st.info("No calculations yet. Use the calculator to add entries!")

# Batch calculation section
st.markdown("---")
st.subheader("📋 Batch CSV Upload")

uploaded_file = st.file_uploader("Upload CSV file with bag measurements", type="csv")

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        
        # Validate columns
        required_cols = ['Description', 'Strap Length', 'Bag Width']
        missing_cols = [col for col in required_cols if col not in batch_df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.markdown("Required format:")
            st.code("""Description,Strap Length,Bag Width
Large Tote,19.0,15.75
Small Purse,12.5,8.0""")
        else:
            st.write("Preview of uploaded data:")
            st.dataframe(batch_df.head())
            
            if st.button("Process Batch", type="primary"):
                results = []
                progress_bar = st.progress(0)
                
                for idx, row in batch_df.iterrows():
                    progress = (idx + 1) / len(batch_df)
                    progress_bar.progress(progress)
                    
                    result_row = {
                        'Description': row['Description'],
                        'Strap Length': row['Strap Length'],
                        'Bag Width': row['Bag Width']
                    }
                    
                    # Calculate for different offsets
                    offsets = {
                        "Squared Up": 0,
                        "1 Inch Offset": 1,
                        "2 Inch Offset": 2
                    }
                    
                    for name, offset in offsets.items():
                        effective_width = row['Bag Width'] - offset
                        
                        if effective_width <= 0:
                            result_row[name] = "N/A (width ≤ 0)"
                        elif row['Strap Length'] < effective_width:
                            result_row[name] = "N/A (strap < width)"
                        else:
                            try:
                                drop = calculate_strap_drop_simple(row['Strap Length'], effective_width)
                                result_row[name] = f"{drop:.2f}\""
                            except Exception as e:
                                result_row[name] = "Error"
                    
                    results.append(result_row)
                
                progress_bar.empty()
                
                # Display results
                results_df = pd.DataFrame(results)
                st.success(f"✅ Processed {len(results_df)} bags!")
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Download results
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Batch Results as CSV",
                    data=csv_results,
                    file_name="batch_calculation_results.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Example CSV template
with st.expander("📝 Download Example CSV Template"):
    example_data = {
        'Description': ['Large Tote', 'Medium Shoulder Bag', 'Small Crossbody', 'Clutch with Strap', 'Weekend Bag'],
        'Strap Length': [19.0, 18.75, 16.5, 12.0, 24.0],
        'Bag Width': [15.75, 12.0, 9.25, 6.5, 18.0]
    }
    example_df = pd.DataFrame(example_data)
    example_csv = example_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Example Template",
        data=example_csv,
        file_name="bag_measurements_template.csv",
        mime="text/csv"
    )