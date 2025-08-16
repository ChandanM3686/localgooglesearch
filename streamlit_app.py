import asyncio
import requests
import re
import aiohttp
from urllib.parse import urljoin, urlparse
from flask import Flask
import google.generativeai as genai
import streamlit as st
import json
import pandas as pd
import io
import base64
from datetime import datetime

MAPS_API_KEY = "AIzaSyDrLHe1dTTc78XG7lznF0fWF0o5uKR6HXA"
GENAI_API_KEY = "AIzaSyDyqTVoAQAr3JulygFdYmMoqnQRSgK-8GA"


if MAPS_API_KEY == "YOUR_GOOGLE_MAPS_API_KEY" or GENAI_API_KEY == "YOUR_GEMINI_API_KEY":
    st.error("API keys are not configured. Please paste your actual keys into the streamlit_app.py file.")
    st.stop()

try:
    genai.configure(api_key=GENAI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash") # Using a powerful model for better suggestions
except Exception as e:
    st.error(f"Failed to configure the Generative AI model. Please check your GENAI_API_KEY. Error: {e}")
    st.stop()

app = Flask(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# --- Helper Functions ---
def parse_keywords_from_text(text):
    """Parse keywords from comma or space separated text."""
    if not text:
        return []
    keywords = []
    for item in text.split(','):
        item = item.strip()
        if item:
            if ' ' in item and ',' not in text:
                keywords.extend([k.strip() for k in item.split() if k.strip()])
            else:
                keywords.append(item)
    return [k for k in keywords if k]

def parse_keywords_from_file(uploaded_file):
    """Parse keywords from uploaded file."""
    try:
        if uploaded_file.type == "text/plain":
            content = str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "text/csv":
            content = str(uploaded_file.read(), "utf-8")
        else:
            return []
        
        keywords = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                if ',' in line:
                    keywords.extend([k.strip() for k in line.split(',') if k.strip()])
                else:
                    keywords.append(line)
        return keywords
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return []

def ensure_absolute_url(base: str, link: str) -> str:
    """Ensure a URL is absolute, using the base URL if it's relative."""
    if not link:
        return ""
    if base and not urlparse(base).scheme:
        base = "https://" + base
    if urlparse(link).scheme:
        return link
    return urljoin(base.rstrip("/") + "/", link.lstrip("/"))

def domain_is_valid(domain_part: str) -> bool:
    """Check if the domain part of an email address appears valid."""
    if not domain_part:
        return False
    return bool(re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", domain_part))

def run_async(coro):     
    """Helper to run async code from the sync Streamlit environment."""
    return asyncio.run(coro)

# -----------------------------
# Asynchronous Scrapers & AI Functions
# -----------------------------
async def get_pages_to_scan(session, url: str) -> list[str]:
    """Analyzes a website to find the most relevant pages for contact info."""
    prioritized_urls, fallback_urls = set(), set()
    try:
        if not urlparse(url).scheme: url = "https://" + url
        async with session.get(url, timeout=10, headers=HEADERS, ssl=False) as response:
            if response.status != 200: return [url]
            html = await response.text()
            all_links = re.findall(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE)
            for link in all_links:
                if any(kw in link for kw in ["contact", "kontakt"]):
                    prioritized_urls.add(ensure_absolute_url(url, link))
                elif any(kw in link for kw in ["about", "imprint", "legal"]):
                    fallback_urls.add(ensure_absolute_url(url, link))
            if prioritized_urls: return list(prioritized_urls)[:3]
            return list(dict.fromkeys([url] + list(fallback_urls)))[:3]
    except Exception:
        return [url]

async def extract_emails_from_website(session, url):
    if not url: return []
    urls_to_scan = await get_pages_to_scan(session, url)
    all_emails = set()
    for page_url in urls_to_scan:
        try:
            async with session.get(page_url, timeout=10, headers=HEADERS, ssl=False) as resp:
                if resp.status != 200: continue
                html = await resp.text()
                email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
                for e in re.findall(email_pattern, html):
                    if not any(ext in e.lower() for ext in [".png", ".jpg", ".gif", ".webp", ".svg"]):
                        if "@" in e and domain_is_valid(e.split("@")[1]):
                            all_emails.add(e)
        except Exception as e:
            print(f"Could not extract emails from {page_url}: {e}")
    return list(all_emails)


async def extract_social_media(session, url):
    if not url: return {}
    urls_to_scan = await get_pages_to_scan(session, url)
    social_handles = {}
    social_patterns = {
        "Facebook": r'facebook\.com/([\w.\-]+)', "Twitter": r'(?:twitter|x)\.com/([\w_]+)',
        "Instagram": r'instagram\.com/([\w._\-]+)', "LinkedIn": r'linkedin\.com/(?:company|in)/([\w\-]+)',
        "YouTube": r'youtube\.com/(?:user|channel|c)/([\w\-@]+)',
    }
    for page_url in urls_to_scan:
        try:
            async with session.get(page_url, timeout=10, headers=HEADERS, ssl=False) as resp:
                if resp.status != 200: continue
                html = await resp.text()
                for platform, pattern in social_patterns.items():
                    if platform not in social_handles:
                        if match := re.search(pattern, html, re.IGNORECASE):
                            handle = match.group(1)
                            base_url = f"https://www.{platform.lower()}.com/"
                            if platform == "LinkedIn": base_url = "https://www.linkedin.com/company/"
                            social_handles[platform] = ensure_absolute_url(base_url, handle)
        except Exception as e:
            print(f"Could not extract social media from {page_url}: {e}")
    return social_handles

async def analyze_business_nature(company_name, website):
    if not website: return ""
    prompt = f"Describe the business category for '{company_name}' ({website}). Be concise (e.g., 'Italian Restaurant', 'Digital Marketing Agency'). Max 10 words."
    try:
        response = await model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error analyzing business nature: {e}")
        return ""

# --- NEW FEATURE: AI-Powered Target Suggestion ---
async def suggest_targets_with_ai(product_info: str):
    """Uses Gemini to suggest target industries and locations based on a product description."""
    if not product_info:
        return None
    prompt = f"""
    Based on the following product/service description, please suggest potential B2B target markets.
    
    Product/Service Description: "{product_info}"
    
    Please provide your answer in a clean JSON format. The JSON object should have two keys:
    1. "industries": A list of 3-5 specific target industry strings (e.g., "Software Development Companies", "Italian Restaurants", "Dental Clinics").
    2. "locations": A list of 3-5 suitable target locations (e.g., "San Francisco, United States", "London, United Kingdom").
    
    Example Response:
    {{
        "industries": ["Boutique Coffee Shops", "Independent Bookstores", "Artisan Bakeries"],
        "locations": ["Seattle, United States", "Portland, United States", "Melbourne, Australia"]
    }}
    
    Your Response:
    """
    try:
        response = await model.generate_content_async(prompt)
        # Clean up the response to extract only the JSON part
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_text)
    except Exception as e:
        st.error(f"Failed to get AI suggestions. Error: {e}")
        return None

def is_relevant_business(business_name, business_type, keyword):
    """Check if business is relevant to the search keyword."""
    if not business_name or not keyword:
        return False
    
    keyword_lower = keyword.lower().strip()
    business_name_lower = business_name.lower()
    business_type_lower = (business_type or "").lower()
    
    # Direct keyword match in business name or type
    if (keyword_lower in business_name_lower or 
        keyword_lower in business_type_lower or
        any(word in business_name_lower for word in keyword_lower.split()) or
        any(word in business_type_lower for word in keyword_lower.split())):
        return True
    
    # Comprehensive category matching for any product/service
    keyword_categories = {
        # Food & Dining
        'restaurant': ['restaurant', 'dining', 'food', 'eatery', 'bistro', 'cafe', 'kitchen', 'diner', 'grill'],
        'cafe': ['cafe', 'coffee', 'espresso', 'latte', 'cappuccino', 'coffeehouse', 'tea', 'beverage'],
        'bakery': ['bakery', 'bread', 'pastry', 'cake', 'baking', 'patisserie', 'confectionery'],
        
        # Accommodation & Travel
        'hotel': ['hotel', 'inn', 'lodge', 'resort', 'accommodation', 'hospitality', 'motel', 'guest house'],
        'travel': ['travel', 'tour', 'tourism', 'vacation', 'holiday', 'trip', 'booking'],
        
        # Health & Wellness
        'gym': ['gym', 'fitness', 'workout', 'exercise', 'training', 'health club', 'yoga', 'pilates'],
        'clinic': ['clinic', 'medical', 'health', 'doctor', 'physician', 'healthcare', 'hospital', 'dental'],
        'pharmacy': ['pharmacy', 'chemist', 'drugstore', 'medicine', 'pharmaceutical', 'medical store'],
        'spa': ['spa', 'wellness', 'massage', 'therapy', 'relaxation', 'beauty treatment'],
        
        # Technology & Electronics
        'laptop': ['laptop', 'computer', 'notebook', 'pc', 'electronics', 'technology', 'it store'],
        'mobile': ['mobile', 'phone', 'smartphone', 'cell phone', 'electronics', 'gadget'],
        'electronics': ['electronics', 'gadgets', 'appliances', 'devices', 'technology', 'digital'],
        'software': ['software', 'app', 'digital', 'tech', 'it services', 'development'],
        
        # Retail & Shopping
        'store': ['store', 'shop', 'retail', 'market', 'outlet', 'boutique', 'mall', 'shopping'],
        'clothing': ['clothing', 'fashion', 'apparel', 'garments', 'textile', 'boutique', 'dress'],
        'jewelry': ['jewelry', 'jewellery', 'gold', 'silver', 'gems', 'ornaments', 'accessories'],
        'furniture': ['furniture', 'home decor', 'interior', 'furnishing', 'design', 'household'],
        
        # Automotive
        'garage': ['garage', 'auto', 'car', 'vehicle', 'automotive', 'repair', 'service', 'mechanic'],
        'car dealer': ['car dealer', 'automobile', 'vehicle sales', 'auto showroom', 'car sales'],
        
        # Professional Services
        'office': ['office', 'business', 'corporate', 'company', 'firm', 'agency', 'consultant'],
        'bank': ['bank', 'banking', 'financial', 'credit', 'loan', 'atm', 'finance'],
        'law': ['law', 'legal', 'lawyer', 'attorney', 'advocate', 'court', 'legal services'],
        'accounting': ['accounting', 'tax', 'finance', 'bookkeeping', 'audit', 'chartered accountant'],
        
        # Education
        'school': ['school', 'education', 'learning', 'academy', 'institute', 'college', 'university'],
        'training': ['training', 'coaching', 'tutorial', 'skill development', 'course', 'workshop'],
        
        # Beauty & Personal Care
        'salon': ['salon', 'beauty', 'hair', 'spa', 'grooming', 'styling', 'barber', 'parlor'],
        'cosmetics': ['cosmetics', 'beauty products', 'makeup', 'skincare', 'personal care'],
        
        # Real Estate & Construction
        'real estate': ['real estate', 'property', 'housing', 'apartment', 'villa', 'plot', 'builder'],
        'construction': ['construction', 'contractor', 'builder', 'architecture', 'civil', 'engineering'],
        
        # Manufacturing & Industrial
        'manufacturing': ['manufacturing', 'factory', 'production', 'industrial', 'machinery', 'equipment'],
        'textile': ['textile', 'fabric', 'garment', 'clothing manufacturer', 'apparel', 'fashion'],
        
        # Agriculture & Food Processing
        'agriculture': ['agriculture', 'farming', 'crops', 'organic', 'food processing', 'dairy'],
        'organic': ['organic', 'natural', 'eco-friendly', 'sustainable', 'green', 'health food'],
        
        # Entertainment & Media
        'entertainment': ['entertainment', 'event', 'party', 'celebration', 'music', 'dance'],
        'media': ['media', 'advertising', 'marketing', 'digital marketing', 'social media', 'content'],
        
        # Transportation & Logistics
        'logistics': ['logistics', 'transport', 'shipping', 'delivery', 'courier', 'freight'],
        'taxi': ['taxi', 'cab', 'ride', 'transport service', 'car rental', 'vehicle hire']
    }
    
    # Check if keyword matches any category
    for category, related_words in keyword_categories.items():
        if keyword_lower in related_words or category in keyword_lower:
            return any(word in business_name_lower or word in business_type_lower 
                      for word in related_words)
    
    return False

async def fetch_leads_by_keywords(keywords, location, city, locality, search_type, radius):
    """Enhanced search function for multiple keywords and location types."""
    all_results = []
    
    for keyword in keywords:
        if not keyword.strip():
            continue
            
        try:
            # Determine search strategy based on available location data
            if locality and city and location:
                # Locality-specific search
                base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
                search_query = f"{keyword} in {locality}, {city}, {location}"
                params = {"query": search_query, "type": "establishment", "key": MAPS_API_KEY}
                
            elif radius and city and location:
                # Radius-based search
                geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
                geocode_params = {"address": f"{city}, {location}", "key": MAPS_API_KEY}
                geocode_response = requests.get(geocode_url, params=geocode_params, timeout=10)
                
                if geocode_data := geocode_response.json().get("results"):
                    loc = geocode_data[0]["geometry"]["location"]
                    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
                    params = {
                        "location": f"{loc['lat']},{loc['lng']}",
                        "radius": int(float(radius) * 1000),
                        "keyword": keyword,
                        "type": "establishment",
                        "key": MAPS_API_KEY
                    }
                else:
                    # Fallback to text search
                    base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
                    search_query = f"{keyword} in {city}, {location}"
                    params = {"query": search_query, "type": "establishment", "key": MAPS_API_KEY}
                    
            elif city and location:
                # City-wide search
                base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
                search_query = f"{keyword} in {city}, {location}"
                params = {"query": search_query, "type": "establishment", "key": MAPS_API_KEY}
                
            elif city or location:
                # Basic location search
                base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
                search_query = f"{keyword} in {city or location}"
                params = {"query": search_query, "type": "establishment", "key": MAPS_API_KEY}
            else:
                continue
            
            response = requests.get(base_url, params=params, timeout=20)
            response.raise_for_status()
            places = response.json().get("results", [])
            
            # Debug: Show what we're searching for
            st.info(f"🔍 Searching: '{search_query if 'search_query' in locals() else keyword}' - Found {len(places)} places")
            
            # Process places and filter for relevance
            async with aiohttp.ClientSession() as session:
                tasks = [process_single_place(session, place, keyword) for place in places[:15]]
                results = await asyncio.gather(*tasks)
                
                # Filter results for relevance (less strict for debugging)
                relevant_results = []
                for result in results:
                    if result:
                        # For debugging, accept all results initially
                        relevant_results.append(result)
                
                all_results.extend(relevant_results)
                st.success(f"✅ Added {len(relevant_results)} results for '{keyword}'")
                
        except Exception as e:
            st.error(f"Error searching for '{keyword}': {e}")
            continue
    
    # Remove duplicates based on company name and address
    unique_results = []
    seen = set()
    for result in all_results:
        key = (result.get("Company Name", ""), result.get("Address", ""))
        if key not in seen:
            seen.add(key)
            unique_results.append(result)
    
    return unique_results

async def fetch_leads(industry, location, city, search_type, radius):
    search_query = f"{industry} in {city}, {location}" if city and location else f"{industry} in {city or location}"
    
    base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": search_query, "key": MAPS_API_KEY}

    if search_type == "Radius" and radius:
        geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
        geocode_params = {"address": f"{city}, {location}" if city else location, "key": MAPS_API_KEY}
        try:
            geocode_response = requests.get(geocode_url, params=geocode_params, timeout=10)
            if geocode_data := geocode_response.json().get("results"):
                loc = geocode_data[0]["geometry"]["location"]
                base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
                params = {
                    "location": f"{loc['lat']},{loc['lng']}", "radius": int(float(radius) * 1000),
                    "keyword": industry, "key": MAPS_API_KEY
                }
        except Exception as e:
            print(f"Geocoding failed, falling back to text search. Error: {e}")
    
    response = requests.get(base_url, params=params, timeout=20)
    response.raise_for_status()
    places = response.json().get("results", [])

    async with aiohttp.ClientSession() as session:
        tasks = [process_single_place(session, place, industry) for place in places[:15]]
        return [lead for lead in await asyncio.gather(*tasks) if lead]

async def process_single_place(session, place, industry):
    place_id = place.get("place_id")
    if not place_id: return None

    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
    details_params = {
        "place_id": place_id, "fields": "name,formatted_address,website,formatted_phone_number",
        "key": MAPS_API_KEY
    }
    try:
        async with session.get(details_url, params=details_params, timeout=10, ssl=False) as resp:
            details = (await resp.json()).get("result", {})
    except Exception: return None

    website = details.get("website")
    company_name = details.get("name", "")
    
    emails, social_handles, business_nature = [], {}, industry
    if website:
        tasks = [
            extract_emails_from_website(session, website),
            extract_social_media(session, website),
            analyze_business_nature(company_name, website)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        emails = results[0] if not isinstance(results[0], Exception) else []
        social_handles = results[1] if not isinstance(results[1], Exception) else {}
        business_nature = results[2] if not isinstance(results[2], Exception) else industry

    return {
        "Company Name": company_name, "Nature of Business": business_nature,
        "Email IDs": emails, "Contact Numbers": details.get("formatted_phone_number", ""),
        "Social Media Handles": social_handles, "Address": details.get("formatted_address", ""),
        "Website": website or "",
    }

# -----------------------------
# Streamlit App UI
# -----------------------------
def main():
    st.set_page_config(
        page_title="Local Data Search Module",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Local Data Search Module")
    st.markdown("Extract business information within specified geographic areas using keyword-based search")

    # Simplified Input Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Keywords Input
        st.subheader("Keywords")
        keyword_method = st.radio(
            "Input method:",
            ["Type keywords", "Upload file"],
            horizontal=True
        )
        
        keywords = []
        if keyword_method == "Type keywords":
            keyword_text = st.text_area(
                "Enter your product details:",
                placeholder="Digital marketing services, Web development, Mobile app development, SEO consulting",
                height=80
            )
            keywords = parse_keywords_from_text(keyword_text)
            
        else:  # Upload file
            uploaded_file = st.file_uploader(
                "Upload CSV/TXT file:",
                type=['txt', 'csv'],
                help="Download sample_keywords.csv for format reference"
            )
            if uploaded_file:
                keywords = parse_keywords_from_file(uploaded_file)
        
        # AI Suggestions (simplified)
        with st.expander("🤖 AI Suggestions (Optional)"):
            product_info = st.text_area("Describe your product/service:", height=80)
            if st.button("Get AI Suggestions"):
                if product_info:
                    with st.spinner("Getting suggestions..."):
                        suggestions = run_async(suggest_targets_with_ai(product_info))
                        st.session_state['suggestions'] = suggestions
                else:
                    st.warning("Please describe your product/service first.")
            
            if 'suggestions' in st.session_state and st.session_state['suggestions']:
                suggestions = st.session_state['suggestions']
                st.write("**Industries:** " + ", ".join(suggestions.get("industries", ["N/A"])))
                st.write("**Locations:** " + ", ".join(suggestions.get("locations", ["N/A"])))

        # Target Industry Input
        industry = st.text_input("🎯 Enter your target industry:", placeholder="e.g., Restaurants, Medical Clinics, Retail Stores, Law Firms")
        if industry and not keywords:
            keywords = [industry]
    
    with col2:
        if keywords:
            st.success(f"✅ {len(keywords)} keywords")
            with st.expander("View keywords"):
                for i, kw in enumerate(keywords, 1):
                    st.write(f"{i}. {kw}")
        else:
            st.info("No keywords loaded")

    # Location Section
    st.subheader("Location")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        location = st.text_input("Country:", placeholder="India")
    
    with col2:
        city = st.text_input("City:", placeholder="Mumbai")
    
    with col3:
        radius = st.slider("Radius (km):", 1, 50, 10)
    
    with col4:
        locality = st.text_input("Locality:", placeholder="Bandra West")

    # Search Button
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        search_button = st.button(
            "🔍 Search Businesses",
            type="primary",
            use_container_width=True,
            disabled=not (keywords and (location or city))
        )
    
    with search_col2:
        if st.button("Clear"):
            if 'search_results' in st.session_state:
                del st.session_state['search_results']
            st.rerun()

    # Search Logic
    if search_button:
        if not keywords:
            st.error("❌ Please provide at least one keyword")
        elif not (location or city):
            st.error("❌ Please enter Country and City")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔍 Searching for businesses...")
                progress_bar.progress(25)
            
                # Use new multi-keyword search function
                results = run_async(fetch_leads_by_keywords(
                    keywords, location, city, locality, None, radius
                ))
                
                progress_bar.progress(75)
                status_text.text("📊 Processing results...")
                
                st.session_state['search_results'] = results
                st.session_state['search_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                progress_bar.progress(100)
                status_text.text("✅ Search completed!")
                
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"❌ Search failed: {e}")
                progress_bar.empty()
                status_text.empty()

    # Results Display and Download
    if 'search_results' in st.session_state:
        results = st.session_state['search_results']
        
        st.header("4️⃣ Search Results")
        
        if not results:
            st.warning("⚠️ No businesses found matching your criteria. Try:")
            st.markdown("- Using broader keywords")
            st.markdown("- Expanding the search radius")
            st.markdown("- Checking the location spelling")
        else:
            # Results summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Results", len(results))
            with col2:
                with_emails = sum(1 for r in results if r.get('Email IDs'))
                st.metric("📧 With Emails", with_emails)
            with col3:
                with_phones = sum(1 for r in results if r.get('Contact Numbers'))
                st.metric("📞 With Phones", with_phones)
            
            # Download options
            st.subheader("📥 Download Options")
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                # CSV Download
                df = pd.DataFrame(results)
                df['Social Media Handles'] = df['Social Media Handles'].apply(
                    lambda x: ', '.join([f"{k}: {v}" for k, v in x.items()]) if isinstance(x, dict) else str(x)
                )
                df['Email IDs'] = df['Email IDs'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                )
                
                csv_filename = f"business_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.download_button(
                    label="📊 Download All Results (CSV)",
                    data=df.to_csv(index=False),
                    file_name=csv_filename,
                    mime="text/csv",
                    use_container_width=True
                )
            
            with download_col2:
                # JSON Download
                json_filename = f"business_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                st.download_button(
                    label="📄 Download All Results (JSON)",
                    data=json.dumps(results, indent=2),
                    file_name=json_filename,
                    mime="application/json",
                    use_container_width=True
                )
            
            # Individual results display
            st.subheader("📋 Individual Results")
            
            for i, business in enumerate(results, 1):
                with st.expander(f"🏢 {business.get('Company Name', 'Unknown Business')} - {business.get('Address', 'No address')}"):
                    
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.markdown(f"**🏢 Company:** {business.get('Company Name', 'N/A')}")
                        st.markdown(f"**🏭 Business Type:** {business.get('Nature of Business', 'N/A')}")
                        st.markdown(f"**📍 Address:** {business.get('Address', 'N/A')}")
                        if website := business.get('Website'):
                            st.markdown(f"**🌐 Website:** [{website}]({website})")
                    
                    with detail_col2:
                        st.markdown(f"**📞 Phone:** {business.get('Contact Numbers', 'N/A')}")
                        
                        emails = business.get('Email IDs', [])
                        if emails:
                            st.markdown(f"**📧 Emails:** {', '.join(emails)}")
                        else:
                            st.markdown("**📧 Emails:** Not found")
                        
                        social_handles = business.get('Social Media Handles', {})
                        if social_handles:
                            social_links = [f"[{platform}]({url})" for platform, url in social_handles.items()]
                            st.markdown(f"**📱 Social Media:** {' | '.join(social_links)}")
                        else:
                            st.markdown("**📱 Social Media:** Not found")
                    
                    # Individual download button
                    individual_json = json.dumps(business, indent=2)
                    st.download_button(
                        label=f"📥 Download {business.get('Company Name', 'Business')} Data (JSON)",
                        data=individual_json,
                        file_name=f"{business.get('Company Name', 'business').replace(' ', '_')}_data.json",
                        mime="application/json",
                        key=f"download_{i}"
                    )

if __name__ == "__main__":
    main()
