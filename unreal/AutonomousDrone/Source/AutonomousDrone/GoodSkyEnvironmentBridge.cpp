// Copyright Epic Games, Inc. All Rights Reserved.

#include "CoreMinimal.h"
#include "Components/DirectionalLightComponent.h"
#include "Components/MeshComponent.h"
#include "Components/SkyLightComponent.h"
#include "Components/SkyAtmosphereComponent.h"
#include "Components/VolumetricCloudComponent.h"
#include "Engine/DirectionalLight.h"
#include "Engine/Engine.h"
#include "Engine/SkyLight.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "HAL/IConsoleManager.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "UObject/SoftObjectPath.h"
#include "UObject/UnrealType.h"

namespace GoodSkyEnvironmentBridge
{
namespace
{
FString NormalizeName(const FString& Value)
{
	FString Result;
	for (const TCHAR Character : Value)
	{
		if (FChar::IsAlnum(Character))
		{
			Result.AppendChar(FChar::ToLower(Character));
		}
	}
	return Result;
}

FString PropertyLabel(const FProperty* Property)
{
	return Property ? Property->GetDisplayNameText().ToString() : FString();
}

bool PropertyMatches(const FProperty* Property, const TArray<FString>& Candidates)
{
	if (!Property)
	{
		return false;
	}

	const FString PropertyName = NormalizeName(Property->GetName());
	const FString DisplayName = NormalizeName(PropertyLabel(Property));
	for (const FString& Candidate : Candidates)
	{
		const FString NormalizedCandidate = NormalizeName(Candidate);
		if (PropertyName == NormalizedCandidate || DisplayName == NormalizedCandidate)
		{
			return true;
		}
	}
	return false;
}

FProperty* FindProperty(UObject* Object, const TArray<FString>& Candidates)
{
	if (!Object)
	{
		return nullptr;
	}

	for (TFieldIterator<FProperty> It(Object->GetClass(), EFieldIterationFlags::IncludeSuper); It; ++It)
	{
		if (PropertyMatches(*It, Candidates))
		{
			return *It;
		}
	}
	return nullptr;
}

bool SetBool(UObject* Object, const TArray<FString>& Candidates, bool Value)
{
	if (FBoolProperty* Property = CastField<FBoolProperty>(FindProperty(Object, Candidates)))
	{
		Property->SetPropertyValue_InContainer(Object, Value);
		return true;
	}
	return false;
}

bool SetNumber(UObject* Object, const TArray<FString>& Candidates, double Value)
{
	FProperty* Property = FindProperty(Object, Candidates);
	if (FFloatProperty* FloatProperty = CastField<FFloatProperty>(Property))
	{
		FloatProperty->SetPropertyValue_InContainer(Object, static_cast<float>(Value));
		return true;
	}
	if (FDoubleProperty* DoubleProperty = CastField<FDoubleProperty>(Property))
	{
		DoubleProperty->SetPropertyValue_InContainer(Object, Value);
		return true;
	}
	if (FIntProperty* IntProperty = CastField<FIntProperty>(Property))
	{
		IntProperty->SetPropertyValue_InContainer(Object, FMath::RoundToInt(Value));
		return true;
	}
	return false;
}

bool SetObject(UObject* Object, const TArray<FString>& Candidates, UObject* Value)
{
	if (FObjectPropertyBase* Property = CastField<FObjectPropertyBase>(FindProperty(Object, Candidates)))
	{
		if (!Value || Value->IsA(Property->PropertyClass))
		{
			Property->SetObjectPropertyValue_InContainer(Object, Value);
			return true;
		}
	}
	return false;
}

int64 FindEnumValue(const UEnum* Enum, const TArray<FString>& DesiredLabels)
{
	if (!Enum)
	{
		return INDEX_NONE;
	}

	for (const FString& DesiredLabel : DesiredLabels)
	{
		const FString Desired = NormalizeName(DesiredLabel);
		for (int32 Index = 0; Index < Enum->NumEnums(); ++Index)
		{
			if (Enum->HasMetaData(TEXT("Hidden"), Index))
			{
				continue;
			}
			const FString Internal = NormalizeName(Enum->GetNameStringByIndex(Index));
			const FString Display = NormalizeName(Enum->GetDisplayNameTextByIndex(Index).ToString());
			if (Internal.Contains(Desired) || Display.Contains(Desired))
			{
				return Enum->GetValueByIndex(Index);
			}
		}
	}
	return INDEX_NONE;
}

bool SetEnum(
	UObject* Object,
	const TArray<FString>& Candidates,
	const TArray<FString>& DesiredLabels,
	FString* AppliedLabel = nullptr)
{
	FProperty* Property = FindProperty(Object, Candidates);
	if (FEnumProperty* EnumProperty = CastField<FEnumProperty>(Property))
	{
		const int64 Value = FindEnumValue(EnumProperty->GetEnum(), DesiredLabels);
		if (Value != INDEX_NONE)
		{
			void* Address = EnumProperty->ContainerPtrToValuePtr<void>(Object);
			EnumProperty->GetUnderlyingProperty()->SetIntPropertyValue(Address, Value);
			if (AppliedLabel)
			{
				*AppliedLabel = EnumProperty->GetEnum()->GetDisplayNameTextByValue(Value).ToString();
			}
			return true;
		}
	}
	if (FByteProperty* ByteProperty = CastField<FByteProperty>(Property))
	{
		const int64 Value = FindEnumValue(ByteProperty->Enum, DesiredLabels);
		if (Value != INDEX_NONE)
		{
			ByteProperty->SetPropertyValue_InContainer(Object, static_cast<uint8>(Value));
			if (AppliedLabel)
			{
				*AppliedLabel = ByteProperty->Enum->GetDisplayNameTextByValue(Value).ToString();
			}
			return true;
		}
	}
	return false;
}

FString GetEnumLabel(UObject* Object, const TArray<FString>& Candidates)
{
	FProperty* Property = FindProperty(Object, Candidates);
	if (FEnumProperty* EnumProperty = CastField<FEnumProperty>(Property))
	{
		const void* Address = EnumProperty->ContainerPtrToValuePtr<void>(Object);
		const int64 Value = EnumProperty->GetUnderlyingProperty()->GetSignedIntPropertyValue(Address);
		return EnumProperty->GetEnum()->GetDisplayNameTextByValue(Value).ToString();
	}
	if (FByteProperty* ByteProperty = CastField<FByteProperty>(Property))
	{
		const uint8 Value = ByteProperty->GetPropertyValue_InContainer(Object);
		return ByteProperty->Enum
			? ByteProperty->Enum->GetDisplayNameTextByValue(Value).ToString()
			: FString::FromInt(Value);
	}
	return TEXT("unknown");
}

bool InvokeNoArgumentFunctions(UObject* Object, const TArray<FString>& CandidateNames)
{
	if (!Object)
	{
		return false;
	}

	bool bInvoked = false;
	for (TFieldIterator<UFunction> It(Object->GetClass(), EFieldIterationFlags::IncludeSuper); It; ++It)
	{
		UFunction* Function = *It;
		if (!Function || Function->ParmsSize != 0)
		{
			continue;
		}
		const FString FunctionName = NormalizeName(Function->GetName());
		for (const FString& Candidate : CandidateNames)
		{
			if (FunctionName == NormalizeName(Candidate))
			{
				Object->ProcessEvent(Function, nullptr);
				bInvoked = true;
				break;
			}
		}
	}
	return bInvoked;
}

AActor* FindOrSpawnGoodSky(UWorld* World)
{
	if (!World)
	{
		return nullptr;
	}

	const FSoftClassPath GoodSkyClassPath(TEXT("/Game/GoodSky/Blueprint/BP_GoodSky.BP_GoodSky_C"));
	UClass* GoodSkyClass = GoodSkyClassPath.TryLoadClass<AActor>();
	if (!GoodSkyClass)
	{
		UE_LOG(LogTemp, Warning, TEXT("DroneEnv: BP_GoodSky is not installed."));
		return nullptr;
	}

	for (TActorIterator<AActor> It(World, GoodSkyClass); It; ++It)
	{
		return *It;
	}

	FActorSpawnParameters SpawnParameters;
	SpawnParameters.Name = TEXT("DroneRuntime_GoodSky");
	SpawnParameters.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
	return World->SpawnActor<AActor>(GoodSkyClass, FVector::ZeroVector, FRotator::ZeroRotator, SpawnParameters);
}

int32 RestoreSkyAtmosphereAndConfigureClouds(UWorld* World, bool bShowClouds)
{
	// Good SKY's sphere material does not render a lower hemisphere.  If the
	// level SkyAtmosphere is hidden, that empty part is exposed as a solid black
	// band below the horizon.  Keep SkyAtmosphere as the background fill while
	// Good SKY continues to provide the visible sky, sun, moon and presets.
	for (TActorIterator<ASkyAtmosphere> It(World); It; ++It)
	{
		It->SetActorHiddenInGame(false);
		if (USkyAtmosphereComponent* Component = It->GetComponent())
		{
			Component->SetVisibility(true, true);
			Component->MarkRenderStateDirty();
		}
	}

	int32 CloudCount = 0;
	for (TActorIterator<AVolumetricCloud> It(World); It; ++It)
	{
		It->SetActorHiddenInGame(!bShowClouds);
		if (UVolumetricCloudComponent* Component = It->FindComponentByClass<UVolumetricCloudComponent>())
		{
			Component->SetVisibility(bShowClouds, true);
			Component->MarkRenderStateDirty();
		}
		++CloudCount;
	}
	return CloudCount;
}

void ApplyTimeLighting(
	UWorld* World,
	ADirectionalLight* DirectionalLight,
	bool bMorning,
	bool bEvening,
	bool bMidnight,
	float& OutSunIntensity,
	float& OutSkyIntensity)
{
	OutSunIntensity = bMidnight ? 0.01f : (bEvening ? 2.0f : (bMorning ? 5.0f : 10.0f));
	// Use a strong midnight ambient fill so RGB collection preserves visible
	// building, person, bird and vehicle detail while the sky remains nocturnal.
	OutSkyIntensity = bMidnight ? 0.5f : (bEvening ? 0.35f : (bMorning ? 0.65f : 1.0f));

	if (DirectionalLight)
	{
		if (UDirectionalLightComponent* Component = DirectionalLight->GetComponent())
		{
			Component->SetIntensity(OutSunIntensity);
			Component->MarkRenderStateDirty();
		}
	}
	for (TActorIterator<ASkyLight> It(World); It; ++It)
	{
		if (USkyLightComponent* Component = It->GetLightComponent())
		{
			Component->SetIntensity(OutSkyIntensity);
			Component->MarkRenderStateDirty();
		}
	}
}

int32 ApplySunDiscSize(
	AActor* GoodSky,
	bool bMorning,
	bool bEvening,
	bool bMidnight,
	float& OutSelfRadius,
	float& OutGlowRadius)
{
	// The SunSetSmoothly table row uses a deliberately stylized, very large
	// sunset disc (Self Radius 0.0025, Glow Radius 0.7).  Use a restrained disc
	// only for the evening collection preset and restore the normal preset-sized
	// values whenever the operator selects another time of day.
	OutSelfRadius = bEvening ? 0.00030f : 0.00250f;
	OutGlowRadius = bEvening ? 0.30f : (bMorning ? 0.30f : 0.50f);
	if (bMidnight)
	{
		OutGlowRadius = 0.30f;
	}

	int32 UpdatedMaterialCount = 0;
	TArray<UMeshComponent*> MeshComponents;
	GoodSky->GetComponents<UMeshComponent>(MeshComponents);
	for (UMeshComponent* Mesh : MeshComponents)
	{
		if (!Mesh)
		{
			continue;
		}
		for (int32 MaterialIndex = 0; MaterialIndex < Mesh->GetNumMaterials(); ++MaterialIndex)
		{
			if (UMaterialInstanceDynamic* Material = Mesh->CreateDynamicMaterialInstance(MaterialIndex))
			{
				Material->SetScalarParameterValue(TEXT("Sun Self Radius"), OutSelfRadius);
				Material->SetScalarParameterValue(TEXT("Sun Glow Radius"), OutGlowRadius);
				++UpdatedMaterialCount;
			}
		}
	}
	return UpdatedMaterialCount;
}

void ApplyGoodSky(const TArray<FString>& Args, UWorld* World)
{
	if (!World || !World->IsGameWorld())
	{
		return;
	}

	const double TimeOfDayHours = Args.Num() > 0 ? FCString::Atod(*Args[0]) : 13.0;
	const FString Visibility = Args.Num() > 1 ? Args[1].ToLower() : TEXT("clear");
	const FString Precipitation = Args.Num() > 2 ? Args[2].ToLower() : TEXT("none");
	const bool bMidnight = TimeOfDayHours < 5.0 || TimeOfDayHours >= 21.0;
	const bool bMorning = TimeOfDayHours >= 5.0 && TimeOfDayHours < 11.0;
	const bool bEvening = TimeOfDayHours >= 16.0 && TimeOfDayHours < 21.0;
	const bool bStorm = Precipitation == TEXT("rain") || Precipitation == TEXT("snow");
	const bool bShowVolumetricClouds = Visibility == TEXT("cloudy") || bStorm;
	AActor* GoodSky = FindOrSpawnGoodSky(World);
	if (!GoodSky)
	{
		if (GEngine)
		{
			GEngine->AddOnScreenDebugMessage(
				-1, 10.0f, FColor::Red,
				TEXT("ENV ERROR: BP_GoodSky was not found or could not be spawned."));
		}
		return;
	}
	if (GEngine)
	{
		GEngine->AddOnScreenDebugMessage(
			-1,
			8.0f,
			FColor::Cyan,
			FString::Printf(
				TEXT("ENV REQUEST: %.1fh | visibility=%s | precipitation=%s"),
				TimeOfDayHours,
				*Visibility,
				*Precipitation));
	}
	const int32 LegacyCloudCount = RestoreSkyAtmosphereAndConfigureClouds(
		World, bShowVolumetricClouds);

	// Good SKY needs its time-of-day graph enabled even when a fixed visual
	// preset is selected. Keep the graph enabled, but disable automatic cycling
	// so the requested collection condition remains deterministic.
	const bool bTimeModeEnabled = SetBool(GoodSky, {TEXT("Enable Time Of Day")}, true);
	SetBool(GoodSky, {TEXT("Enable Auto Day Night Cycle In Game")}, false);
	SetBool(GoodSky, {TEXT("Use Random Time For Custom Mode"), TEXT("Random Time")}, false);
	SetBool(GoodSky, {TEXT("Use All Random Sky"), TEXT("Use All Random")}, false);
	FString AppliedSkyMesh;
	const bool bSkyMeshApplied = SetEnum(
		GoodSky,
		{TEXT("SkyMesh"), TEXT("Sky Mesh")},
		{TEXT("Sphere")},
		&AppliedSkyMesh);
	SetNumber(GoodSky, {TEXT("Get Present Time Of Day"), TEXT("Present Time Of Day")}, TimeOfDayHours);
	SetNumber(GoodSky, {TEXT("Temp Time Of Day"), TEXT("Time Of Day")}, TimeOfDayHours);
	TArray<FString> PresetLabels;
	if (bMidnight)
	{
		PresetLabels = bStorm
			? TArray<FString>{TEXT("Midnight Storm")}
			: TArray<FString>{TEXT("Midnight Moon"), TEXT("Midnight Stars")};
	}
	else if (bMorning)
	{
		PresetLabels = {TEXT("SunRise From Time Of Day"), TEXT("Style UDK Morning Sky")};
	}
	else if (bEvening)
	{
		PresetLabels = {TEXT("SunSet Smoothly"), TEXT("SunSet From Time Of Day"), TEXT("SunSet In Burn")};
	}
	else
	{
		PresetLabels = {TEXT("Noon Clear Sky"), TEXT("Noon From Time Of Day")};
	}
	FString AppliedPreset;
	const bool bPresetApplied = SetEnum(
		GoodSky,
		{TEXT("SkyPreset"), TEXT("Sky Preset")},
		PresetLabels,
		&AppliedPreset);

	SetEnum(
		GoodSky,
		{TEXT("SkyEffect"), TEXT("Sky Effect")},
		bStorm ? TArray<FString>{TEXT("Storm")}
		       : TArray<FString>{TEXT("Sun Stars Moon"), TEXT("Sun Stars")});

	TArray<FString> CoverageLabels;
	if (Visibility == TEXT("clear"))
	{
		CoverageLabels = {TEXT("Clear")};
	}
	else if (Visibility == TEXT("cloudy"))
	{
		CoverageLabels = {TEXT("Super Heavy"), TEXT("Middle")};
	}
	else
	{
		CoverageLabels = {TEXT("Middle"), TEXT("Slight")};
	}
	FString AppliedCoverage;
	const bool bCoverageApplied = SetEnum(
		GoodSky,
		{TEXT("SkyCloudsCoveragePreset"), TEXT("Sky Clouds Coverage Preset")},
		CoverageLabels,
		&AppliedCoverage);

	ADirectionalLight* DirectionalLight = nullptr;
	for (TActorIterator<ADirectionalLight> It(World); It; ++It)
	{
		DirectionalLight = *It;
		break;
	}
	if (DirectionalLight)
	{
		SetObject(
			GoodSky,
			{TEXT("DirectionalLight"), TEXT("Direction Light Actor"), TEXT("Light Actor")},
			DirectionalLight);
	}

	// SpawnActor already runs the construction script once. Re-running the full
	// Good SKY construction graph for every button press can synchronously rebuild
	// materials and freeze PIE, especially when switching back from midnight.
	// Its lightweight realtime update event is sufficient for later preset changes.
	const bool bRefreshInvoked = InvokeNoArgumentFunctions(
		GoodSky,
		{TEXT("GoodSky Realtime Update"),
		 TEXT("Refresh Sky Shader For Direction Actor"),
		 TEXT("Update Direction Light Actor")});
	GoodSky->MarkComponentsRenderStateDirty();
	float SunSelfRadius = 0.0f;
	float SunGlowRadius = 0.0f;
	const int32 SunMaterialCount = ApplySunDiscSize(
		GoodSky,
		bMorning,
		bEvening,
		bMidnight,
		SunSelfRadius,
		SunGlowRadius);
	float SunIntensity = 0.0f;
	float SkyIntensity = 0.0f;
	ApplyTimeLighting(
		World,
		DirectionalLight,
		bMorning,
		bEvening,
		bMidnight,
		SunIntensity,
		SkyIntensity);
	const FString ActualPreset = GetEnumLabel(
		GoodSky, {TEXT("SkyPreset"), TEXT("Sky Preset")});
	const FString ActualCoverage = GetEnumLabel(
		GoodSky, {TEXT("SkyCloudsCoveragePreset"), TEXT("Sky Clouds Coverage Preset")});
	const FString ActualSkyMesh = GetEnumLabel(
		GoodSky, {TEXT("SkyMesh"), TEXT("Sky Mesh")});
	const bool bApplied =
		bPresetApplied && bCoverageApplied && bTimeModeEnabled && bSkyMeshApplied;
	if (GEngine)
	{
		GEngine->AddOnScreenDebugMessage(
			-1,
			10.0f,
			bApplied ? FColor::Green : FColor::Red,
			FString::Printf(
				TEXT("ENV RESULT: preset=%s | clouds=%s | sky-mesh=%s | refresh=%s | sky-atmosphere=fill | volumetric-cloud=%s (%d actor) | sun-radius=%.5f glow=%.2f (%d material) | sun=%.3f | skylight=%.3f | %s"),
				*ActualPreset,
				*ActualCoverage,
				*ActualSkyMesh,
				bRefreshInvoked ? TEXT("called") : TEXT("not found"),
				bShowVolumetricClouds ? TEXT("visible") : TEXT("hidden"),
				LegacyCloudCount,
				SunSelfRadius,
				SunGlowRadius,
				SunMaterialCount,
				SunIntensity,
				SkyIntensity,
				bApplied ? TEXT("PROPERTY OK") : TEXT("PROPERTY FAILED")));
	}

	UE_LOG(
		LogTemp,
		Display,
		TEXT("DroneEnv: Good SKY applied (time %.1f, visibility %s, precipitation %s, preset=%s [%s], coverage=%s [%s], sky-mesh=%s [%s], time-mode-on=%s, actual-preset=%s, actual-coverage=%s, actual-sky-mesh=%s, refresh=%s, sky-atmosphere=fill, volumetric-cloud=%s (%d actor), sun-radius=%.5f, sun-glow=%.2f (%d material), sun=%.3f, skylight=%.3f)."),
		TimeOfDayHours,
		*Visibility,
		*Precipitation,
		*AppliedPreset,
		bPresetApplied ? TEXT("ok") : TEXT("failed"),
		*AppliedCoverage,
		bCoverageApplied ? TEXT("ok") : TEXT("failed"),
		*AppliedSkyMesh,
		bSkyMeshApplied ? TEXT("ok") : TEXT("failed"),
		bTimeModeEnabled ? TEXT("ok") : TEXT("failed"),
		*ActualPreset,
		*ActualCoverage,
		*ActualSkyMesh,
		bRefreshInvoked ? TEXT("called") : TEXT("not-found"),
		bShowVolumetricClouds ? TEXT("visible") : TEXT("hidden"),
		LegacyCloudCount,
		SunSelfRadius,
		SunGlowRadius,
		SunMaterialCount,
		SunIntensity,
		SkyIntensity);
}

FAutoConsoleCommandWithWorldAndArgs ApplyGoodSkyCommand(
	TEXT("DroneEnv.ApplyGoodSky"),
	TEXT("Apply Mission Control time and cloud preset to BP_GoodSky."),
	FConsoleCommandWithWorldAndArgsDelegate::CreateStatic(&ApplyGoodSky));
} // namespace
} // namespace GoodSkyEnvironmentBridge
