import React from 'react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { X } from "lucide-react";

interface FilterOptions {
  industries: string[];
  stages: string[];
  locations: string[];
}

interface SelectedFilters {
  industry: string;
  stage: string;
  location: string;
}

interface CompanyFiltersProps {
  options: FilterOptions;
  selected: SelectedFilters;
  onChange: (name: string, value: string) => void;
}

const CompanyFilters: React.FC<CompanyFiltersProps> = ({
  options,
  selected,
  onChange
}) => {
  const selectStyle = "bg-gray-800 border border-gray-700 rounded-lg py-2 px-4 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 appearance-none w-full md:w-auto";

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const { name, value } = e.target;
    onChange(name, value);
  };

  return (
    <div className="flex flex-col md:flex-row space-y-3 md:space-y-0 md:space-x-3 mb-4 p-4 bg-gray-800 border border-gray-700 rounded-lg">
      <div className="flex-1">
        <label htmlFor="industry" className="block text-sm text-gray-400 mb-1">業界</label>
        <select
          id="industry"
          name="industry"
          className={selectStyle}
          value={selected.industry}
          onChange={handleChange}
        >
          <option value="">すべての業界</option>
          {options.industries.map(industry => (
            <option key={industry} value={industry}>{industry}</option>
          ))}
        </select>
      </div>

      <div className="flex-1">
        <label htmlFor="stage" className="block text-sm text-gray-400 mb-1">ステージ</label>
        <select
          id="stage"
          name="stage"
          className={selectStyle}
          value={selected.stage}
          onChange={handleChange}
        >
          <option value="">すべてのステージ</option>
          {options.stages.map(stage => (
            <option key={stage} value={stage}>{stage}</option>
          ))}
        </select>
      </div>

      <div className="flex-1">
        <label htmlFor="location" className="block text-sm text-gray-400 mb-1">所在地</label>
        <select
          id="location"
          name="location"
          className={selectStyle}
          value={selected.location}
          onChange={handleChange}
        >
          <option value="">すべての所在地</option>
          {options.locations.map(location => (
            <option key={location} value={location}>{location}</option>
          ))}
        </select>
      </div>
    </div>
  );
};

export default CompanyFilters;