import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="أمانة منطقة الجوف", layout="centered")

st.markdown("""
    <style>
    * {
        direction: rtl;
        text-align: right;
        font-family: 'Cairo', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)
st.sidebar.image("big_logo.png", use_container_width=True)
st.sidebar.markdown("""
##  نظام التنبؤ الذكي لإصدار الرخص التجارية في منطقة الجوف """)

st.sidebar.markdown("""
### نبذة عن المشروع  
يهدف هذا النظام إلى تقديم استشارات ذكية للمستخدمين حول أفضل الأوقات لتقديم طلبات إصدار الرخص التجارية، بناءً على تحليل البيانات التاريخية والتنبؤ بالطلب المستقبلي.""")

st.sidebar.markdown("""### كيفية الاستخدام  
1. اختر تاريخ تقديم الطلب من التقويم.
2. أدخل تفاصيل الطلب مثل البلدية، وفئة مدة صلاحية الرخصة.
3. اضغط على زر "تحليل الطلب" للحصول على الاستشارة المناسبة.
""")
st.sidebar.markdown("""
                    ### البيانات المستخدمة
                    تم استخدام ملف بيانات الرخص التجارية من أمانة منطقة الجوف، والذي يحتوي على معلومات حول الطلبات المقدمة، أنواع الرخص، والأنشطة التجارية المختلفة.
                    """)


# Load models and encoders
@st.cache_resource
def load_models():
    with open('prophet_model.pkl', 'rb') as f:
        prophet_model = joblib.load(f)
    with open('type_cvModel.pkl', 'rb') as f:
        type_model = joblib.load(f)
    with open('act_cvModel.pkl', 'rb') as f:
        activity_model = joblib.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = joblib.load(f)
    return prophet_model, type_model, activity_model, encoders

prophet_model, type_model, activity_model, encoders = load_models()

# Define advisory function
def advise_user(future_date, prophet_model, type_model, activity_model, encoders, user_input, threshold=500):
    date = pd.to_datetime(future_date)
    month = date.month
    week = date.isocalendar().week
    day = date.day

    # Forecast demand
    last_train_date = prophet_model.history['ds'].max()
    days_needed = (date - last_train_date).days + 30
    days_needed = abs(days_needed)
    future_df = prophet_model.make_future_dataframe(periods=days_needed, freq='D')
    forecast = prophet_model.predict(future_df)
    predicted_row = forecast[forecast['ds'] == date]

    if predicted_row.empty:
        return "! لا توجد بيانات تنبؤية لهذا التاريخ."

    predicted_demand = predicted_row['yhat'].values[0]

    # Build input features
    features = {
        'البلدية': user_input['البلدية'],
        'فئة مدة صلاحية الرخصة': user_input['فئة مدة صلاحية الرخصة'],
        'شهر الطلب': month,
        'أسبوع الطلب': week,
        'يوم الطلب': day
    }

    # Encode input
    X_encoded = {}
    for col, val in features.items():
        if col in encoders:
            try:
                X_encoded[col] = encoders[col].transform([val])[0]
            except ValueError:
                return f"⚠️ القيمة '{val}' غير معروفة في العمود '{col}'"
        else:
            X_encoded[col] = val

    input_df = pd.DataFrame([X_encoded])

    # Predict and decode
    encoded_type = type_model.predict(input_df)[0]
    encoded_activity = activity_model.predict(input_df)[0]
    predicted_type = encoders['نوع الطلب'].inverse_transform([encoded_type])[0]
    predicted_activity = encoders['اسم النشاط الرئيسي'].inverse_transform([encoded_activity])[0]

    # Final advisory message
    if predicted_demand > threshold:
        return f"""
📅 التاريخ: {date.strftime('%d %B %Y')}  
⚠️ الموسم الحالي مزدحم. يُفضل تأجيل التقديم لتجنّب التأخير.  
🔹 الطلبات الشائعة: {predicted_type}  
🔹 الأنشطة السائدة: {predicted_activity}
""".strip()
    else:
        return f"""
📅 التاريخ: {date.strftime('%d %B %Y')}  
✅ الوقت مناسب لتقديم الطلب، لا يوجد ضغط مرتفع على النظام.  
🔹 الطلبات المتوقعة: {predicted_type}  
🔹 الأنشطة الأكثر شيوعاً: {predicted_activity}
""".strip()
# Main app content
st.title("نظام التنبؤ الذكي لإصدار الرخص التجارية")

# Date input
date_input = st.date_input("اختر تاريخ تقديم الطلب:")

# User inputs
col1, col2 = st.columns(2)

with col1:
    municipality = st.selectbox("اختر البلدية:", encoders['البلدية'].classes_)
    validity_category = st.selectbox("مدة صلاحية الرخصة:", encoders['فئة مدة صلاحية الرخصة'].classes_)

# Predict button
if st.button("تحليل الطلب"):
    user_input = {
        'البلدية': municipality,
        'فئة مدة صلاحية الرخصة': validity_category
    }

    with st.spinner('جارٍ تحليل البيانات...'):
        result = advise_user(date_input, prophet_model, type_model, activity_model, encoders, user_input)

    st.markdown(result)
