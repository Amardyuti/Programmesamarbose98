<?xml version="1.0" encoding="utf-8"?> 
<resources> 
	<color name="colorPrimary">#42C14B</color> 
	<color name="colorPrimaryDark">#3BC545</color> 
	<color name="colorAccent">#05af9b</color> 

	<color name="color_one">#fb7268</color> 
	<color name="color_white">#ededf2</color> 

</resources> 
// Volley library 
implementation 'com.android.volley:volley:1.1.1' 
<!--Allow Internet Permission-->
<uses-permission android:name="android.permission.INTERNET" /> 
<?xml version="1.0" encoding="utf-8"?> 
<ScrollView
	xmlns:android="http://schemas.android.com/apk/res/android"
	xmlns:app="http://schemas.android.com/apk/res-auto"
	xmlns:tools="http://schemas.android.com/tools"
	android:layout_width="match_parent"
	android:layout_height="match_parent"
	android:background="@color/color_white"
	android:orientation="vertical"
	tools:context=".MainActivity"> 


	<!--Linear Layout to display all the details-->
	<LinearLayout
		android:layout_width="match_parent"
		android:layout_height="wrap_content"
		android:padding="20dp"
		android:orientation="vertical"> 

		<!--Text view to display Global stats-->
		<TextView
			android:layout_width="match_parent"
			android:layout_height="wrap_content"
			android:text="Global Stats"
			android:textColor="#050505"
			android:textAllCaps="true"
			android:textAlignment="center"
			android:textSize="35sp"
			android:textStyle="bold"/> 

			<!--Text view to display Total Cases-->
			<TextView
				android:layout_width="match_parent"
				android:layout_height="wrap_content"
				android:fontFamily="sans-serif-light"
				android:layout_marginTop="20dp"
				android:text="Total Cases"
				android:textAlignment="center"
				android:textStyle="bold"
				android:textSize="30sp"/> 

			<!--Text view to display the updated number when data 
			will fetch from the API. For now default set to 0 -->
			<TextView
				android:layout_width="match_parent"
				android:layout_height="wrap_content"
				android:text="0"
				android:id="@+id/tvCases"
				android:textAlignment="center"
				android:textSize="25sp"
				android:textColor="@color/color_one"
				android:textStyle="bold"
				android:fontFamily="sans-serif-light" /> 

		<!--Text view to display Recovered Cases-->
		<TextView
			android:layout_width="match_parent"
			android:layout_height="wrap_content"
			android:fontFamily="sans-serif-light"
			android:text="Recovered"
			android:textAlignment="center"
			android:textStyle="bold"
			android:textSize="30sp"/> 

		<!--Text view to display the updated number when data 
		will fetch from the API. For now default set to 0 -->
		<TextView
			android:layout_width="match_parent"
			android:layout_height="wrap_content"
			android:text="0"
			android:id="@+id/tvRecovered"
			android:textAlignment="center"
			android:textSize="25sp"
			android:textColor="@color/color_one"
			android:textStyle="bold"
			android:fontFamily="sans-serif-light" /> 

		<!--Text view to display Critical Cases-->
		<TextView
			android:layout_width="match_parent"
			android:layout_height="wrap_content"
			android:fontFamily="sans-serif-light"
			android:text="Critical"
			android:textAlignment="center"
			android:textStyle="bold"
			android:textSize="30sp"/> 

		<!--Text view to display the updated number when data 
		will fetch from the API. For now default set to 0 -->
		<TextView
			android:layout_width="match_parent"
			android:layout_height="wrap_content"
			android:text="0"
			android:id="@+id/tvCritical"
			android:textAlignment="center"
			android:textSize="25sp"
			android:textColor="@color/color_one"
			android:textStyle="bold"
			android:fontFamily="sans-serif-light" /> 

		<!--Text view to display Active Cases-->
		<TextView
			android:layout_width="match_parent"
			android:layout_height="wrap_content"
			android:fontFamily="sans-serif-light"
			android:text="Active"
			android:textAlignment="center"
			android:textStyle="bold"
			android:textSize="30sp"/> 

		<!--Text view to display the updated number when data 
		will fetch from the API. For now default set to 0 -->
		<TextView
			android:layout_width="match_parent"
			android:layout_height="wrap_content"
			android:text="0"
			android:id="@+id/tvActive"
			android:textAlignment="center"
			android:textSize="25sp"
			android:textColor="@color/color_one"
			android:textStyle="bold"
			android:fontFamily="sans-serif-light" /> 

		<!--Text view to display Today Cases-->
		<TextView
			android:layout_width="match_parent"
			android:layout_height="wrap_content"
			android:fontFamily="sans-serif-light"
			android:text="Today Cases"
			android:textAlignment="center"
			android:textStyle="bold"
			android:textSize="30sp"/> 

		<!--Text view to display the updated number when data 
		will fetch from the API. For now default set to 0 -->
		<TextView
			android:layout_width="match_parent"
			android:layout_height="wrap_content"
			android:text="0"
			android:id="@+id/tvTodayCases"
			android:textAlignment="center"
			android:textSize="25sp"
			android:textColor="@color/color_one"
			android:textStyle="bold"
			android:fontFamily="sans-serif-light" /> 

		<!--Text view to display Total Deaths-->
		<TextView
			android:layout_width="match_parent"
			android:layout_height="wrap_content"
			android:fontFamily="sans-serif-light"
			android:text="Total Deaths"
			android:textAlignment="center"
			android:textStyle="bold"
			android:textSize="30sp"/> 

		<!--Text view to display the updated number when data 
		will fetch from the API. For now default set to 0 -->
		<TextView
			android:layout_width="match_parent"
			android:layout_height="wrap_content"
			android:text="0"
			android:id="@+id/tvTotalDeaths"
			android:textAlignment="center"
			android:textSize="25sp"
			android:textColor="@color/color_one"
			android:textStyle="bold"
			android:fontFamily="sans-serif-light" /> 

		<!--Text view to display Today Deaths-->
		<TextView
			android:layout_width="match_parent"
			android:layout_height="wrap_content"
			android:fontFamily="sans-serif-light"
			android:text="Today Deaths"
			android:textAlignment="center"
			android:textStyle="bold"
			android:textSize="30sp"/> 

		<!--Text view to display the updated number when data 
		will fetch from the API. For now default set to 0 -->
		<TextView
			android:layout_width="match_parent"
			android:layout_height="wrap_content"
			android:text="0"
			android:id="@+id/tvTodayDeaths"
			android:textAlignment="center"
			android:textSize="25sp"
			android:textColor="@color/color_one"
			android:textStyle="bold"
			android:fontFamily="sans-serif-light" /> 

		<!--Text view to display Affected Countries-->
		<TextView
			android:layout_width="match_parent"
			android:layout_height="wrap_content"
			android:fontFamily="sans-serif-light"
			android:text="Affected Countries"
			android:textAlignment="center"
			android:textStyle="bold"
			android:textSize="30sp"/> 

		<!--Text view to display the updated number when data 
		will fetch from the API. For now default set to 0 -->
		<TextView
			android:layout_width="match_parent"
			android:layout_height="wrap_content"
			android:text="0"
			android:id="@+id/tvAffectedCountries"
			android:textAlignment="center"
			android:textSize="25sp"
			android:textColor="@color/color_one"
			android:textStyle="bold"
			android:fontFamily="sans-serif-light" /> 

		</LinearLayout> 

</ScrollView> 
// Create a String request using Volley Library 

String url = "https:// corona.lmao.ninja/v2/all"; 

StringRequest request 
	= new StringRequest( 
		Request.Method.GET, 
		url, 
		new Response.Listener() { 
			@Override
			public void onResponse( 
				String response) 
			{ 
			} 
		}, 
		new Response.ErrorListener() { 
			@Override
			public void onErrorResponse( 
				VolleyError error) 
			{ 
			} 
		}); 

RequestQueue requestQueue 
	= Volley.newRequestQueue(this); 
requestQueue.add(request); 
// Handle the JSON object and handle it inside try and catch 

try { 
	// Creating object of JSONObject 
	JSONObject jsonObject 
		= new JSONObject( 
			response.toString()); 

	// Set the data in text view 
	// which are available in JSON format 
	// Note that the parameter 
	// inside the getString() must match 
	// with the name given in JSON format 
	tvCases.setText( 
		jsonObject.getString("cases")); 
	tvRecovered.setText( 
		jsonObject.getString("recovered")); 
	tvCritical.setText( 
		jsonObject.getString("critical")); 
	tvActive.setText( 
		jsonObject.getString("active")); 
	tvTodayCases.setText( 
		jsonObject.getString("todayCases")); 
	tvTotalDeaths.setText( 
		jsonObject.getString("deaths")); 
	tvTodayDeaths.setText( 
		jsonObject.getString("todayDeaths")); 
	tvAffectedCountries.setText( 
		jsonObject.getString("affectedCountries")); 
} 
catch (JSONException e) { 
	e.printStackTrace(); 
} 
package com.example.covid_19tracker; 

import androidx.appcompat.app.AppCompatActivity; 

import android.os.Bundle; 
import android.widget.TextView; 
import android.widget.Toast; 

import com.android.volley.Request; 
import com.android.volley.RequestQueue; 
import com.android.volley.Response; 
import com.android.volley.VolleyError; 
import com.android.volley.toolbox.StringRequest; 
import com.android.volley.toolbox.Volley; 

import org.json.JSONException; 
import org.json.JSONObject; 

public class MainActivity 
	extends AppCompatActivity { 

	// Create the object of TextView 
	TextView tvCases, tvRecovered, 
		tvCritical, tvActive, 
		tvTodayCases, tvTotalDeaths, 
		tvTodayDeaths, 
		tvAffectedCountries; 

	@Override
	protected void onCreate(Bundle savedInstanceState) 
	{ 
		super.onCreate(savedInstanceState); 
		setContentView(R.layout.activity_main); 

		// Link those objects with their respective id's 
		// that we have given in .XML file 
		tvCases 
			= findViewById(R.id.tvCases); 
		tvRecovered 
			= findViewById(R.id.tvRecovered); 
		tvCritical 
			= findViewById(R.id.tvCritical); 
		tvActive 
			= findViewById(R.id.tvActive); 
		tvTodayCases 
			= findViewById(R.id.tvTodayCases); 
		tvTotalDeaths 
			= findViewById(R.id.tvTotalDeaths); 
		tvTodayDeaths 
			= findViewById(R.id.tvTodayDeaths); 
		tvAffectedCountries 
			= findViewById(R.id.tvAffectedCountries); 

		// Creating a method fetchdata() 
		fetchdata(); 
	} 

	private void fetchdata() 
	{ 

		// Create a String request 
		// using Volley Library 
		String url = "https:// corona.lmao.ninja/v2/all"; 

		StringRequest request 
			= new StringRequest( 
				Request.Method.GET, 
				url, 
				new Response.Listener<String>() { 
					@Override
					public void onResponse(String response) 
					{ 

						// Handle the JSON object and 
						// handle it inside try and catch 
						try { 

							// Creating object of JSONObject 
							JSONObject jsonObject 
								= new JSONObject( 
									response.toString()); 

							// Set the data in text view 
							// which are available in JSON format 
							// Note that the parameter inside 
							// the getString() must match 
							// with the name given in JSON format 
							tvCases.setText( 
								jsonObject.getString( 
									"cases")); 
							tvRecovered.setText( 
								jsonObject.getString( 
									"recovered")); 
							tvCritical.setText( 
								jsonObject.getString( 
									"critical")); 
							tvActive.setText( 
								jsonObject.getString( 
									"active")); 
							tvTodayCases.setText( 
								jsonObject.getString( 
									"todayCases")); 
							tvTotalDeaths.setText( 
								jsonObject.getString( 
									"deaths")); 
							tvTodayDeaths.setText( 
								jsonObject.getString( 
									"todayDeaths")); 
							tvAffectedCountries.setText( 
								jsonObject.getString( 
									"affectedCountries")); 
						} 
						catch (JSONException e) { 
							e.printStackTrace(); 
						} 
					} 
				}, 
				new Response.ErrorListener() { 
					@Override
					public void onErrorResponse(VolleyError error) 
					{ 
						Toast.makeText( 
								MainActivity.this, 
								error.getMessage(), 
								Toast.LENGTH_SHORT) 
							.show(); 
					} 
				}); 

		RequestQueue requestQueue 
			= Volley.newRequestQueue(this); 
		requestQueue.add(request); 
	} 
} 
